import csv
import json
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, width, modes_y, modes_x):
        super().__init__()
        self.my, self.mx = modes_y, modes_x
        shape = (width, width, modes_y, modes_x)
        self.positive = nn.Parameter(torch.randn(*shape, dtype=torch.cfloat) / width)
        self.negative = nn.Parameter(torch.randn(*shape, dtype=torch.cfloat) / width)

    def forward(self, x):
        z = torch.fft.rfft2(x, norm="ortho")
        my = min(self.my, x.shape[-2] // 2)
        mx = min(self.mx, z.shape[-1])
        out = z.new_zeros(z.shape)
        out[:, :, :my, :mx] = torch.einsum(
            "bihw,iohw->bohw", z[:, :, :my, :mx], self.positive[:, :, :my, :mx])
        out[:, :, -my:, :mx] = torch.einsum(
            "bihw,iohw->bohw", z[:, :, -my:, :mx], self.negative[:, :, :my, :mx])
        return torch.fft.irfft2(out, s=x.shape[-2:], norm="ortho")


class FNO(nn.Module):
    def __init__(self, width=32, layers=4, modes_y=20, modes_x=32, padding=12):
        super().__init__()
        self.padding = padding
        # u_n, v_n, x, y, cylinder mask, Re, next inlet speed, dt
        self.lift = nn.Conv2d(8, width, 1)
        self.spectral = nn.ModuleList([
            SpectralConv2d(width, modes_y, modes_x) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.head = nn.Sequential(nn.Conv2d(width, 64, 1), nn.GELU(), nn.Conv2d(64, 3, 1))

    def forward(self, x):
        h, w = x.shape[-2:]
        x = self.lift(x)
        if self.padding:
            x = F.pad(x, (0, self.padding, 0, self.padding))
        for spectral, local in zip(self.spectral, self.local):
            x = F.gelu(spectral(x) + local(x))
        return self.head(x[:, :, :h, :w])


def dx(q, h):
    return torch.cat(((q[..., 1:2] - q[..., :1]) / h,
                      (q[..., 2:] - q[..., :-2]) / (2 * h),
                      (q[..., -1:] - q[..., -2:-1]) / h), dim=-1)


def dy(q, h):
    return torch.cat(((q[..., 1:2, :] - q[..., :1, :]) / h,
                      (q[..., 2:, :] - q[..., :-2, :]) / (2 * h),
                      (q[..., -1:, :] - q[..., -2:-1, :]) / h), dim=-2)


def lap(q, hx, hy):
    return ((q[..., 1:-1, 2:] - 2*q[..., 1:-1, 1:-1] + q[..., 1:-1, :-2]) / hx**2
            + (q[..., 2:, 1:-1] - 2*q[..., 1:-1, 1:-1] + q[..., :-2, 1:-1]) / hy**2)


class CylinderPhysics:
    def __init__(self, config, device):
        self.c, self.device = config, device
        g = config["grid"]
        x = torch.linspace(g["xmin"], g["xmax"], g["nx"], device=device)
        y = torch.linspace(g["ymin"], g["ymax"], g["ny"], device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        self.x, self.y = xx[None, None], yy[None, None]
        self.hx = (g["xmax"] - g["xmin"]) / (g["nx"] - 1)
        self.hy = (g["ymax"] - g["ymin"]) / (g["ny"] - 1)
        self.radius = torch.sqrt(self.x**2 + self.y**2)
        eps = config["physics"].get("interface_width")
        if eps is None:
            eps = config["physics"]["interface_cells"] * max(self.hx, self.hy)
        self.interface_width = float(eps)
        self.chi = 0.5 * (1 - torch.tanh((self.radius - 0.5) / eps))
        self.eta = config["physics"]["eta"]
        self.free = torch.ones_like(self.x)
        self.free[..., 0, :] = 0
        self.free[..., -1, :] = 0
        self.free[..., :, 0] = 0
        self.weight = 1 + config["physics"]["wake_weight"] * (
            (self.x > 0.5) & (self.x < 10) & (self.y.abs() < 2)).float()

    def inlet(self, time):
        tau = self.c["physics"]["ramp_time"]
        return 1 - math.exp(-(time / tau)**2)

    def initial(self, amplitudes):
        r2 = ((self.x - 2.0) / 0.65)**2 + (self.y / 0.65)**2
        psi = torch.clamp(1 - r2, min=0)**4
        vel = torch.cat((dy(psi, self.hy), -dx(psi, self.hx)), dim=1)
        return amplitudes[:, None, None, None] * vel

    def predict(self, model, previous, re, time):
        dt = self.c["physics"]["dt"]
        b = previous.shape[0]
        shape = (b, 1, previous.shape[-2], previous.shape[-1])
        g = self.c["grid"]
        coords = [(2*(self.x-g["xmin"])/(g["xmax"]-g["xmin"])-1).expand(shape),
                  (2*(self.y-g["ymin"])/(g["ymax"]-g["ymin"])-1).expand(shape)]
        speed = self.inlet(time + dt)
        inputs = torch.cat([previous, *coords, self.chi.expand(shape),
                            (re[:, None, None, None]/100).expand(shape),
                            torch.full(shape, speed, device=self.device),
                            torch.full(shape, dt, device=self.device)], dim=1)
        raw = model(inputs)

        velocity = (previous + dt*raw[:, :2]) / (1 + dt*self.chi/self.eta)
        prescribed = torch.cat((torch.full_like(self.x, speed), torch.zeros_like(self.x)), dim=1)
        velocity = velocity*self.free + prescribed*(1-self.free)
        pressure = raw[:, 2:3] - raw[:, 2:3, :, -1:]
        return velocity, pressure

    def residual(self, previous, velocity, pressure, re):
        dt = self.c["physics"]["dt"]
        u, v = velocity[:, :1], velocity[:, 1:2]
        ux, uy = dx(u, self.hx), dy(u, self.hy)
        vx, vy = dx(v, self.hx), dy(v, self.hy)
        gradient = torch.cat((dx(pressure, self.hx), dy(pressure, self.hy)), dim=1)
        advection = torch.cat((u*ux + v*uy, u*vx + v*vy), dim=1)
        momentum = ((velocity-previous)/dt + advection + gradient
                    + self.chi/self.eta*velocity)[..., 1:-1, 1:-1]
        momentum = momentum - lap(velocity, self.hx, self.hy)/re[:, None, None, None]
        divergence = (ux + vy)[..., 1:-1, 1:-1]
        outlet = (velocity[..., 1:-1, -1] - velocity[..., 1:-1, -2]) / self.hx
        reduce = lambda a: a.square().flatten(1).mean(1)
        solid = (self.radius < 0.4).float()
        solid_mse = (velocity.square()*solid).flatten(1).sum(1)/(2*solid.sum()).clamp_min(1)
        terms = {"momentum": reduce(momentum), "divergence": reduce(divergence),
                 "outlet": reduce(outlet), "solid": solid_mse}
        w = self.c["loss"]
        loss = (w["momentum"] * (momentum.square()*self.weight[..., 1:-1, 1:-1]).mean()
                + w["divergence"]*terms["divergence"].mean()
                + w["outlet"]*terms["outlet"].mean()
                + w["solid"]*solid_mse.mean())
        metrics = {k: float(value.detach().max().sqrt().cpu()) for k, value in terms.items()}
        return loss, metrics

    def accepted(self, metrics):
        return all(math.isfinite(v) and v <= self.c["tolerances"][k] for k, v in metrics.items())

    @torch.no_grad()
    def diagnostics(self, velocity):
        force = (self.chi/self.eta*velocity).sum(dim=(-2, -1))*self.hx*self.hy
        curl = dx(velocity[:, 1:2], self.hx) - dy(velocity[:, :1], self.hy)
        return 2*force[:, 0], 2*force[:, 1], curl


def get_device(name):
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Install a CUDA-enabled PyTorch build and check the NVIDIA driver. No automatic CPU fallback.")
    return device


def validate(c):
    g, p, tr = c["grid"], c["physics"], c["training"]
    if min(g["nx"], g["ny"]) < 17 or not (g["xmin"] < -1 and g["xmax"] > 3 and g["ymin"] < -1 and g["ymax"] > 1):
        raise ValueError("Domain must contain the cylinder and initial perturbation; each grid dimension >=17.")
    if min(p["dt"], p["eta"], p["ramp_time"], p["interface_cells"]) <= 0:
        raise ValueError("Physical scale parameters must be positive.")
    cases = c["cases"]
    if not cases["re"] or len(cases["re"]) != len(cases["amplitude"]) or min(cases["re"]) <= 0:
        raise ValueError("Provide equally sized positive Re and perturbation amplitude lists.")
    if min(tr["steps"], tr["iterations_per_step"], tr["check_every"], tr["save_every"]) < 1 or tr["replay_capacity"] < 0:
        raise ValueError("Invalid training counts.")
    for k in ("momentum", "divergence", "outlet", "solid"):
        if c["tolerances"][k] <= 0 or c["loss"][k] <= 0:
            raise ValueError("Loss weights and tolerances must be positive.")
    width = p.get("interface_width")
    if width is not None and (not math.isfinite(width) or width <= 0):
        raise ValueError("interface_width must be positive or None.")
    d = c.get("diagnostics", {})
    points = d.get("surface_points", 512)
    if not isinstance(points, int) or points < 32 or points % 2:
        raise ValueError("surface_points must be an even integer >=32.")
    if min(d.get("fluid_offset_widths", 3.0), d.get("radial_spacing_cells", 1.0)) <= 0:
        raise ValueError("Positive force sampling offsets required.")


def append_csv(path, row):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def checkpoint(path, model, optimizer, c, state, committed, replay, status):
    payload = {"config": c, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "state": state.detach().cpu(), "committed_steps": committed,
               "replay": replay, "status": status}
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


@torch.no_grad()
def record(out, physics, state, pressure, re, time, step, save_field, previous=None):
    cd, cl, omega = physics.diagnostics(state)
    force_values = force_diagnostics(physics, state, pressure, re, previous)
    if not (out / "force_method.json").exists():
        write_json(out / "force_method.json", {
            "method": "symmetric Cauchy stress, fluid-side linear reconstruction to nominal radius 0.5",
            "interface_width": physics.interface_width,
            "diagnostics": physics.c.get("diagnostics", {}),
            "normalization": "rho=U_ref=D=1; coefficients=2*force per unit span",
            "note": "Cd_total/Cl_total are reconstructed surface estimates, not calibrated exact forces."
        })
    for i in range(len(re)):
        append_csv(out / "force_history.csv", {"step": step, "time": time, "case": i,
                   "Re": float(re[i]), **{key: float(value[i]) for key, value in force_values.items()}})
        append_csv(out / "history.csv", {"step": step, "time": time, "case": i,
                   "Re": float(re[i]), "Cd_penalty_indicator": float(cd[i]),
                   "Cl_penalty_indicator": float(cl[i]),
                   "max_speed": float(state[i].square().sum(0).sqrt().max())})
    if save_field:
        np.savez_compressed(out / ("field_%06d.npz" % step),
                            velocity=state.cpu().numpy(), pressure=pressure.cpu().numpy(),
                            vorticity=omega.cpu().numpy(), re=re.cpu().numpy(), time=time,
                            x=physics.x[0, 0].cpu().numpy(), y=physics.y[0, 0].cpu().numpy())


@torch.no_grad()
def force_diagnostics(physics, velocity, pressure, reynolds, previous=None):
    """Surface-stress estimates, plus sensitivity and diffuse-volume indicators.

    Three fluid-side rings: linear extrapolation of p and velocity gradients to R.
    This is postprocessing of a regularized solution, NOT an exact body-fitted traction.
    All derivatives/interpolation/reductions here use float64; training stays float32.
    """
    c, d = physics.c, physics.c.get("diagnostics", {})
    n = d.get("surface_points", 512)
    radius = 0.5
    spacing = d.get("radial_spacing_cells", 1.0)*max(physics.hx, physics.hy)
    offset = max(d.get("fluid_offset_widths", 3.0)*physics.interface_width, 2*max(physics.hx, physics.hy))
    outer = radius+offset+2*spacing
    g = c["grid"]
    if outer >= min(-g["xmin"], g["xmax"], -g["ymin"], g["ymax"]):
        raise ValueError("Force sampling rings do not fit in the domain.")
    v, p = velocity.double(), pressure.double()
    theta = torch.arange(n, device=v.device, dtype=torch.float64)*(2*math.pi/n)
    nx, ny = torch.cos(theta), torch.sin(theta)
    radii = torch.tensor([radius+offset+i*spacing for i in range(3)], device=v.device, dtype=torch.float64)
    xx, yy = radii[:, None]*nx, radii[:, None]*ny
    grid = torch.stack((2*(xx-g["xmin"])/(g["xmax"]-g["xmin"])-1,
                        2*(yy-g["ymin"])/(g["ymax"]-g["ymin"])-1), dim=-1)
    fields = torch.cat((p, dx(v[:, :1], physics.hx), dy(v[:, :1], physics.hy),
                        dx(v[:, 1:2], physics.hx), dy(v[:, 1:2], physics.hy)), dim=1)
    sampled = F.grid_sample(fields, grid[None].expand(v.shape[0], -1, -1, -1),
                            mode="bilinear", padding_mode="border", align_corners=True)
    primary = sampled[:, :, 0]-offset/spacing*(sampled[:, :, 1]-sampled[:, :, 0])
    shifted = sampled[:, :, 1]-(offset+spacing)/spacing*(sampled[:, :, 2]-sampled[:, :, 1])
    viscosity = 1/reynolds.double().reshape(-1, 1)

    def integrate(values, stride=1):
        pp, ux, uy, vx, vy = (values[:, i, ::stride] for i in range(5))
        normal_x, normal_y = nx[::stride], ny[::stride]
        ds = radius*2*math.pi/(n//stride)
        pressure_x = 2*(-pp*normal_x).sum(-1)*ds
        pressure_y = 2*(-pp*normal_y).sum(-1)*ds
        viscous_x = 2*(viscosity*(2*ux*normal_x+(uy+vx)*normal_y)).sum(-1)*ds
        viscous_y = 2*(viscosity*((uy+vx)*normal_x+2*vy*normal_y)).sum(-1)*ds
        return pressure_x, pressure_y, viscous_x, viscous_y

    dp, lp, dv, lv = integrate(primary)
    dp2, lp2, dv2, lv2 = integrate(shifted)
    dpc, lpc, dvc, lvc = integrate(primary, 2)
    coeff = {"Cd_pressure": dp, "Cl_pressure": lp, "Cd_viscous": dv, "Cl_viscous": lv,
             "Cd_total": dp+dv, "Cl_total": lp+lv,
             "Cd_shifted_estimate": dp2+dv2, "Cl_shifted_estimate": lp2+lv2,
             "Cd_ring_sensitivity": (dp+dv-dp2-dv2).abs(), "Cl_ring_sensitivity": (lp+lv-lp2-lv2).abs(),
             "Cd_quadrature_difference": (dp+dv-dpc-dvc).abs(), "Cl_quadrature_difference": (lp+lv-lpc-lvc).abs(),
             "Cp_front_minus_back": 2*(primary[:, 0, n//2]-primary[:, 0, 0])}

    area_weights = torch.ones_like(physics.chi, dtype=torch.float64)
    area_weights[..., 0, :] *= 0.5
    area_weights[..., -1, :] *= 0.5
    area_weights[..., :, 0] *= 0.5
    area_weights[..., :, -1] *= 0.5
    solid_weights = area_weights*physics.chi.double()*physics.hx*physics.hy
    penalty = 2*(solid_weights*v/physics.eta).sum((-2, -1))
    if previous is None:
        inertia = torch.full_like(penalty, float("nan"))
    else:
        inertia = 2*(solid_weights*(v-previous.double())/c["physics"]["dt"]).sum((-2, -1))
    coeff.update(Cd_penalty_trapezoid=penalty[:, 0], Cl_penalty_trapezoid=penalty[:, 1],
                 Cd_diffuse_inertia=inertia[:, 0], Cl_diffuse_inertia=inertia[:, 1],
                 Cd_penalty_inertia_indicator=penalty[:, 0]+inertia[:, 0],
                 Cl_penalty_inertia_indicator=penalty[:, 1]+inertia[:, 1])
    return coeff


