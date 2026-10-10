
import csv
import json
import math
import inspect
import os
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

VERSION = "equal_area_geometry_v1"


def shape_name(n):
    return "circle" if int(n) == 0 else f"polygon_{int(n)}"


def vertices(n):
    """D_eq=1 => A=pi/4. CCW vertices; first edge is horizontal at the top."""
    n = int(n)
    if n < 3:
        raise ValueError("vertices requires n>=3; 0 denotes a circle elsewhere")
    radius = math.sqrt(math.pi/(2*n*math.sin(2*math.pi/n)))
    angles = math.pi/2-math.pi/n+2*math.pi*np.arange(n)/n
    return radius*np.stack((np.cos(angles), np.sin(angles)), axis=1)


def boundary_points(n, count):
    """Boundary points, outward normals, integration lengths; no polygon vertex normals."""
    if int(n) == 0:
        a = 2*math.pi*np.arange(count)/count
        normal = np.stack((np.cos(a), np.sin(a)), axis=1)
        return .5*normal, normal, np.full(count, math.pi/count)
    vv = vertices(n); edge = np.roll(vv, -1, axis=0)-vv
    length = np.linalg.norm(edge, axis=1)
    normal = np.stack((edge[:, 1], -edge[:, 0]), axis=1)/length[:, None]
    per_edge = max(2, math.ceil(count/int(n)))
    s = (np.arange(per_edge)+.5)/per_edge
    points = (vv[:, None, :]+edge[:, None, :]*s[None, :, None]).reshape(-1, 2)
    return points, np.repeat(normal, per_edge, axis=0), np.repeat(length/per_edge, per_edge)


class SpectralConv2d(nn.Module):
    def __init__(self, width, modes_y, modes_x):
        super().__init__()
        self.my, self.mx = modes_y, modes_x
        shape = (width, width, modes_y, modes_x)
        self.positive = nn.Parameter(torch.randn(*shape, dtype=torch.cfloat)/width)
        self.negative = nn.Parameter(torch.randn(*shape, dtype=torch.cfloat)/width)

    def forward(self, x):
        z = torch.fft.rfft2(x, norm="ortho")
        my, mx = min(self.my, x.shape[-2]//2), min(self.mx, z.shape[-1])
        out = z.new_zeros(z.shape)
        out[:, :, :my, :mx] = torch.einsum("bihw,iohw->bohw", z[:, :, :my, :mx], self.positive[:, :, :my, :mx])
        out[:, :, -my:, :mx] = torch.einsum("bihw,iohw->bohw", z[:, :, -my:, :mx], self.negative[:, :, :my, :mx])
        return torch.fft.irfft2(out, s=x.shape[-2:], norm="ortho")


class FNO(nn.Module):
    def __init__(self, width=32, layers=4, modes_y=20, modes_x=32, padding=12):
        super().__init__()
        self.padding = padding
        # u_n,v_n,x,y,GEOMETRY MASK,Re/100,next inlet speed,dt.
        # Spatial mask carries shape information; no artificial continuous side-number input.
        self.lift = nn.Conv2d(8, width, 1)
        self.spectral = nn.ModuleList([SpectralConv2d(width, modes_y, modes_x) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.head = nn.Sequential(nn.Conv2d(width, 64, 1), nn.GELU(), nn.Conv2d(64, 3, 1))

    def forward(self, x):
        h, w = x.shape[-2:]; x = self.lift(x)
        if self.padding:
            x = F.pad(x, (0, self.padding, 0, self.padding))
        for spectral, local in zip(self.spectral, self.local):
            x = F.gelu(spectral(x)+local(x))
        return self.head(x[:, :, :h, :w])


def dx(q, h):
    return torch.cat(((q[..., 1:2]-q[..., :1])/h, (q[..., 2:]-q[..., :-2])/(2*h),
                      (q[..., -1:]-q[..., -2:-1])/h), dim=-1)


def dy(q, h):
    return torch.cat(((q[..., 1:2, :]-q[..., :1, :])/h, (q[..., 2:, :]-q[..., :-2, :])/(2*h),
                      (q[..., -1:, :]-q[..., -2:-1, :])/h), dim=-2)


def lap(q, hx, hy):
    return ((q[..., 1:-1, 2:]-2*q[..., 1:-1, 1:-1]+q[..., 1:-1, :-2])/hx**2
            +(q[..., 2:, 1:-1]-2*q[..., 1:-1, 1:-1]+q[..., :-2, 1:-1])/hy**2)


class CylinderPhysics:
    """Name retained for compatibility; geometry is now batched and shape-dependent."""
    def __init__(self, config, device):
        self.c, self.device = config, device
        g = config["grid"]; self.sides = list(config["geometry"]["sides"])
        x = torch.linspace(g["xmin"], g["xmax"], g["nx"], device=device)
        y = torch.linspace(g["ymin"], g["ymax"], g["ny"], device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        self.x, self.y = xx[None, None], yy[None, None]
        self.hx = (g["xmax"]-g["xmin"])/(g["nx"]-1)
        self.hy = (g["ymax"]-g["ymin"])/(g["ny"]-1)
        eps = config["physics"].get("interface_width")
        self.interface_width = float(eps if eps is not None else config["physics"]["interface_cells"]*max(self.hx,self.hy))
        self.sdf = torch.cat([self.signed_distance(n) for n in self.sides], dim=0)
        self.chi = .5*(1-torch.tanh(self.sdf/self.interface_width))
        self.solid = (self.sdf < -.1).float()  # same 0.1 interior offset as old radius<0.4
        if (self.solid.sum((-2,-1)) == 0).any():
            raise ValueError("Grid too coarse for a geometry's solid interior")
        self.eta = config["physics"]["eta"]
        self.free = torch.ones_like(self.x)
        self.free[..., 0, :] = 0; self.free[..., -1, :] = 0; self.free[..., :, 0] = 0
        self.weight = 1+config["physics"]["wake_weight"]*((self.x>.5)&(self.x<10)&(self.y.abs()<2)).float()

    def signed_distance(self, n):
        if n == 0:
            return torch.sqrt(self.x**2+self.y**2)-.5
        vv = vertices(n); distance2 = torch.full_like(self.x, float("inf"))
        inside = torch.ones_like(self.x, dtype=torch.bool)
        for a, b in zip(vv, np.roll(vv, -1, axis=0)):
            ex, ey = b-a; px, py = self.x-float(a[0]), self.y-float(a[1])
            projection = torch.clamp((px*ex+py*ey)/(ex*ex+ey*ey), 0, 1)
            distance2 = torch.minimum(distance2, (px-projection*ex)**2+(py-projection*ey)**2)
            inside &= ex*py-ey*px >= 0
        distance = torch.sqrt(distance2)
        return torch.where(inside, -distance, distance)

    def inlet(self, time):
        return 1-math.exp(-(time/self.c["physics"]["ramp_time"])**2)

    def initial(self, amplitudes):
        r2 = ((self.x-2)/.65)**2+(self.y/.65)**2
        psi = torch.clamp(1-r2, min=0)**4
        return amplitudes[:, None, None, None]*torch.cat((dy(psi,self.hy), -dx(psi,self.hx)),dim=1)

    def reynolds(self):
        return torch.full((len(self.sides),), self.c["physics"]["reynolds"], dtype=torch.float32, device=self.device)

    def predict(self, model, previous, re, time):
        dt = self.c["physics"]["dt"]; b = previous.shape[0]
        if b != len(self.sides):
            raise ValueError("Geometry batch does not match state batch")
        shape = (b,1,previous.shape[-2],previous.shape[-1]); g=self.c["grid"]
        coords = [(2*(self.x-g["xmin"])/(g["xmax"]-g["xmin"])-1).expand(shape),
                  (2*(self.y-g["ymin"])/(g["ymax"]-g["ymin"])-1).expand(shape)]
        speed = self.inlet(time+dt)
        inputs = torch.cat([previous,*coords,self.chi,(re[:,None,None,None]/100).expand(shape),
                            torch.full(shape,speed,device=self.device),torch.full(shape,dt,device=self.device)],dim=1)
        raw = model(inputs)
        velocity = (previous+dt*raw[:,:2])/(1+dt*self.chi/self.eta)
        prescribed = torch.cat((torch.full_like(self.x,speed),torch.zeros_like(self.x)),dim=1)
        velocity = velocity*self.free+prescribed*(1-self.free)
        pressure = raw[:,2:3]-raw[:,2:3,:,-1:]
        return velocity, pressure

    def residual(self, previous, velocity, pressure, re):
        dt = self.c["physics"]["dt"]; u,v=velocity[:,:1],velocity[:,1:2]
        ux,uy,vx,vy=dx(u,self.hx),dy(u,self.hy),dx(v,self.hx),dy(v,self.hy)
        gradient=torch.cat((dx(pressure,self.hx),dy(pressure,self.hy)),dim=1)
        advection=torch.cat((u*ux+v*uy,u*vx+v*vy),dim=1)
        momentum=((velocity-previous)/dt+advection+gradient+self.chi/self.eta*velocity)[...,1:-1,1:-1]
        momentum=momentum-lap(velocity,self.hx,self.hy)/re[:,None,None,None]
        divergence=(ux+vy)[...,1:-1,1:-1]
        outlet=(velocity[...,1:-1,-1]-velocity[...,1:-1,-2])/self.hx
        reduce=lambda a:a.square().flatten(1).mean(1)
        # Each shape has its own denominator, not the sum of the whole geometry batch.
        solid_mse=(velocity.square()*self.solid).flatten(1).sum(1)/(2*self.solid.flatten(1).sum(1)).clamp_min(1)
        terms={"momentum":reduce(momentum),"divergence":reduce(divergence),"outlet":reduce(outlet),"solid":solid_mse}
        w=self.c["loss"]
        loss=(w["momentum"]*(momentum.square()*self.weight[...,1:-1,1:-1]).mean()
              +w["divergence"]*terms["divergence"].mean()+w["outlet"]*terms["outlet"].mean()+w["solid"]*solid_mse.mean())
        return loss,{k:float(v.detach().max().sqrt().cpu()) for k,v in terms.items()}

    def accepted(self, metrics):
        return all(math.isfinite(v) and v<=self.c["tolerances"][k] for k,v in metrics.items())

    @torch.no_grad()
    def diagnostics(self, velocity):
        force=(self.chi/self.eta*velocity).sum((-2,-1))*self.hx*self.hy
        return 2*force[:,0],2*force[:,1],dx(velocity[:,1:2],self.hx)-dy(velocity[:,:1],self.hy)


def get_device(name):
    device=torch.device(name)
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no automatic CPU fallback")
    return device


def validate(c):
    if c.get("version")!=VERSION:
        raise ValueError("This version requires a geometry checkpoint; old cylinder-Re weights are not supported")
    g,p,tr=c["grid"],c["physics"],c["training"]; geometry=c["geometry"]
    if geometry.get("diameter_definition")!="equal_area" or geometry.get("equivalent_diameter")!=1 or geometry.get("orientation")!="top_horizontal_edge":
        raise ValueError("This implementation uses D_eq=1, equal area, and a top horizontal polygon edge")
    sides=geometry["sides"]
    if not sides or any(not isinstance(n,int) or isinstance(n,bool) or n not in (0,) and n<3 for n in sides) or len(set(sides))!=len(sides):
        raise ValueError("Unique sides >=3 required; 0 denotes the circle")
    if len(c["cases"]["amplitude"])!=len(sides) or not all(math.isfinite(a) for a in c["cases"]["amplitude"]):
        raise ValueError("One finite initial amplitude per shape is required")
    if p["reynolds"]!=100:
        raise ValueError("This experiment fixes reference Re=100")
    if min(g["nx"],g["ny"])<17 or not(g["xmin"]<-1 and g["xmax"]>3 and g["ymin"]<-1 and g["ymax"]>1):
        raise ValueError("Invalid domain or grid")
    if min(p[k] for k in ("dt","eta","ramp_time","interface_cells"))<=0:
        raise ValueError("Positive physical scales required")
    if p.get("interface_width") is not None and (not math.isfinite(p["interface_width"]) or p["interface_width"]<=0):
        raise ValueError("Positive interface_width required")
    if min(tr[k] for k in ("steps","iterations_per_step","check_every","save_every"))<1 or tr["replay_capacity"]<0:
        raise ValueError("Invalid training counts")
    if any(c["loss"][k]<=0 or c["tolerances"][k]<=0 for k in ("momentum","divergence","outlet","solid")):
        raise ValueError("Positive losses and gates required")
    d=c["diagnostics"]
    if not isinstance(d["surface_points"],int) or d["surface_points"]<32 or min(d["fluid_offset_widths"],d["radial_spacing_cells"])<=0:
        raise ValueError("Invalid force quadrature parameters")


def load_checkpoint(path):
    options={"map_location":"cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters: options["weights_only"]=True
    saved=torch.load(path,**options); validate(saved["config"])
    return saved


def append_csv(path,row):
    exists=path.exists()
    with path.open("a",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(row))
        if not exists: writer.writeheader()
        writer.writerow(row)


def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf-8")


def checkpoint(path,model,optimizer,c,state,committed,replay,status):
    path=Path(path)
    payload={"config":c,"model":model.state_dict(),"optimizer":optimizer.state_dict(),
             "state":state.detach().cpu(),"committed_steps":committed,"replay":replay,"status":status}
    with tempfile.NamedTemporaryFile(mode="wb",prefix="checkpoint_pending_",suffix=".tmp",dir=path.parent,delete=False) as f:
        tmp=Path(f.name)
        try:
            torch.save(payload,f); f.flush(); os.fsync(f.fileno())
        except Exception as exc:
            raise RuntimeError(f"Write incomplete; do not resume from partial file: {tmp}") from exc
    for attempt in range(10):
        try: tmp.replace(path); return
        except OSError as exc:
            if (not isinstance(exc,PermissionError) and getattr(exc,"winerror",None) not in (5,32,33)) or attempt==9:
                raise OSError(f"Checkpoint replacement failed; complete new file retained: {tmp}; step={committed}") from exc
            print("Checkpoint replacement blocked; retry",attempt+1,flush=True); time.sleep(1)


@torch.no_grad()
def force_diagnostics(physics,velocity,pressure,reynolds,previous=None):
    """Fluid-side stress reconstruction on the actual polygon/circle, not a circle surrogate."""
    d=physics.c["diagnostics"]; g=physics.c["grid"]; u,v=velocity[:,:1].double(),velocity[:,1:2].double()
    q=torch.cat((pressure.double(),dx(u,physics.hx),dy(u,physics.hy),dx(v,physics.hx),dy(v,physics.hy)),dim=1)
    spacing=d["radial_spacing_cells"]*max(physics.hx,physics.hy)
    offset=max(d["fluid_offset_widths"]*physics.interface_width,2*max(physics.hx,physics.hy))
    result=[]
    for i,n in enumerate(physics.sides):
        def integrate(count, shifted=False):
            points,normals,lengths=boundary_points(n,count)
            base=offset+(spacing if shifted else 0)
            xy=np.stack((points+base*normals,points+(base+spacing)*normals))
            if (xy[...,0].min()<g["xmin"] or xy[...,0].max()>g["xmax"] or xy[...,1].min()<g["ymin"] or xy[...,1].max()>g["ymax"]):
                raise ValueError("Force sample points outside domain")
            xy[...,0]=2*(xy[...,0]-g["xmin"])/(g["xmax"]-g["xmin"])-1
            xy[...,1]=2*(xy[...,1]-g["ymin"])/(g["ymax"]-g["ymin"])-1
            grid=torch.as_tensor(xy,device=physics.device,dtype=torch.float64)[None]
            sample=F.grid_sample(q[i:i+1],grid,mode="bilinear",align_corners=True)[0]
            wall=sample[:,0]-base/spacing*(sample[:,1]-sample[:,0])
            p,ux,uy,vx,vy=wall
            normal=torch.as_tensor(normals,device=physics.device,dtype=torch.float64)
            ds=torch.as_tensor(lengths,device=physics.device,dtype=torch.float64)
            nx,ny=normal[:,0],normal[:,1]; nu=1/reynolds[i].double()
            return torch.stack((2*(-p*nx*ds).sum(),2*(-p*ny*ds).sum(),
                                2*nu*((2*ux*nx+(uy+vx)*ny)*ds).sum(),
                                2*nu*(((uy+vx)*nx+2*vy*ny)*ds).sum()))
        main=integrate(d["surface_points"]); shifted=integrate(d["surface_points"],True)
        coarse=integrate(max(16,d["surface_points"]//2))
        total=main[:2]+main[2:]; alternate=shifted[:2]+shifted[2:]; half=coarse[:2]+coarse[2:]
        result.append(torch.cat((main,total,alternate,(total-alternate).abs(),(total-half).abs())))
    values=torch.stack(result)
    names=("Cd_pressure","Cl_pressure","Cd_viscous","Cl_viscous","Cd_total","Cl_total",
           "Cd_shifted_estimate","Cl_shifted_estimate","Cd_offset_sensitivity","Cl_offset_sensitivity",
           "Cd_quadrature_difference","Cl_quadrature_difference")
    out={name:values[:,i] for i,name in enumerate(names)}
    weight=torch.ones_like(physics.chi,dtype=torch.float64)
    weight[...,0,:]*=.5; weight[...,-1,:]*=.5; weight[...,:,0]*=.5; weight[...,:,-1]*=.5
    weight*=physics.chi.double()*physics.hx*physics.hy
    penalty=2*(weight*velocity.double()/physics.eta).sum((-2,-1))
    inertia=torch.full_like(penalty,float("nan")) if previous is None else 2*(weight*(velocity.double()-previous.double())/physics.c["physics"]["dt"]).sum((-2,-1))
    for j,prefix in enumerate(("Cd","Cl")):
        out[prefix+"_penalty_trapezoid"]=penalty[:,j]
        out[prefix+"_diffuse_inertia"]=inertia[:,j]
        out[prefix+"_penalty_inertia_indicator"]=penalty[:,j]+inertia[:,j]
    return out


@torch.no_grad()
def record(out,physics,state,pressure,re,time,step,save_field,previous=None):
    cd,cl,omega=physics.diagnostics(state)
    forces=force_diagnostics(physics,state,pressure,re,previous)
    if not(out/"force_method.json").exists():
        write_json(out/"force_method.json",{"method":"Cauchy stress reconstructed along actual boundary normals",
                   "normalization":"rho=U_ref=D_eq=1; C=2F; equal area pi/4",
                   "note":"Polygon edges use midpoint quadrature, not undefined vertex normals. Diffuse-interface stress estimates; corner accuracy is not guaranteed.",
                   "interface_width":physics.interface_width,"diagnostics":physics.c["diagnostics"]})
    for i,n in enumerate(physics.sides):
        meta={"step":step,"time":time,"case":i,"sides":n,"shape":shape_name(n),"Re":float(re[i])}
        append_csv(out/"history.csv",{**meta,"Cd_penalty_indicator":float(cd[i]),"Cl_penalty_indicator":float(cl[i]),
                                    "max_speed":float(state[i].square().sum(0).sqrt().max())})
        append_csv(out/"force_history.csv",{**meta,**{key:float(value[i]) for key,value in forces.items()}})
        # Dense signed probe velocity at (x/D_eq,y/D_eq)=(2,0), same point as before.
        g=physics.c["grid"]
        grid=state.new_tensor([[[[2*(2-g["xmin"])/(g["xmax"]-g["xmin"])-1,
                                  2*(0-g["ymin"])/(g["ymax"]-g["ymin"])-1]]]])
        probe=F.grid_sample(state[i:i+1,1:2],grid,align_corners=True).item()
        append_csv(out/"probe_history.csv",{**meta,"x_D":2.,"y_D":0.,"vp_U":probe})
    if save_field:
        np.savez_compressed(out/("field_%06d.npz"%step),velocity=state.cpu().numpy(),pressure=pressure.cpu().numpy(),
                            vorticity=omega.cpu().numpy(),re=re.cpu().numpy(),sides=np.array(physics.sides,dtype=int),
                            time=time,x=physics.x[0,0].cpu().numpy(),y=physics.y[0,0].cpu().numpy(),
                            solid_mask=(physics.sdf[:,0]<=0).cpu().numpy(),version=VERSION)
