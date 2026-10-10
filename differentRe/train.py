
from copy import deepcopy
import random
import numpy as np
import torch
import params as settings
from model import FNO, CylinderPhysics, get_device, validate, append_csv, write_json, checkpoint, record


def train():
    """读取 params.py，执行原版因果训练；返回退出码与结果目录。"""
    saved = None
    if settings.RESUME_CHECKPOINT:
        resume_path = settings.resolve_path(settings.RESUME_CHECKPOINT)
        saved = torch.load(resume_path, map_location="cpu")
        c = saved["config"]
        out = resume_path.parent
    else:
        c = deepcopy(settings.CONFIG)
        out = settings.fresh_output_directory(settings.TRAIN_OUTPUT)
    validate(c)
    device = get_device(settings.CONFIG["device"])
    torch.manual_seed(c["seed"])
    np.random.seed(c["seed"])
    random.seed(c["seed"])
    # Float32 including FFT; avoid half-precision FFT size restrictions.
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    physics = CylinderPhysics(c, device)
    model = FNO(**c["model"]).to(device)
    tr = c["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=tr["lr"])
    re = torch.tensor(c["cases"]["re"], device=device)
    amp = torch.tensor(c["cases"]["amplitude"], device=device)
    state = physics.initial(amp)
    replay, committed = [], 0
    if saved is not None:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"].to(device)
        committed, replay = saved["committed_steps"], saved["replay"]
    settings.remember_training_directory(out)
    print("Output directory:", out, flush=True)
    write_json(out / "config_used.json", c)
    print("Device:", device, "| cases:", c["cases"]["re"], "| starting step:", committed, flush=True)
    dt = c["physics"]["dt"]
    # The previous accepted state is fixed during each local implicit solve.
    # Gradients never span the entire trajectory (causal truncated training).
    for step in range(committed, tr["steps"]):
        model.train()
        time = step*dt
        for iteration in range(1, tr["iterations_per_step"] + 1):
            optimizer.zero_grad(set_to_none=True)
            velocity, pressure = physics.predict(model, state, re, time)
            loss, _ = physics.residual(state, velocity, pressure, re)
            if replay:
                old_state, old_time = replay[(step + iteration) % len(replay)]
                old_state = old_state.to(device)
                old_velocity, old_pressure = physics.predict(model, old_state, re, old_time)
                old_loss, _ = physics.residual(old_state, old_velocity, old_pressure, re)
                loss = loss + tr["replay_weight"]*old_loss
            if not torch.isfinite(loss):
                checkpoint(out/"checkpoint.pt", model, optimizer, c, state, step, replay, "nonfinite_loss")
                raise RuntimeError("Non-finite loss; time was not advanced. Reduce learning rate/check scales.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr["clip_grad"], error_if_nonfinite=True)
            optimizer.step()
            if iteration % tr["check_every"] == 0 or iteration == tr["iterations_per_step"]:
                with torch.no_grad():
                    velocity, pressure = physics.predict(model, state, re, time)
                    _, metrics = physics.residual(state, velocity, pressure, re)
                append_csv(out / "loss.csv", {"step": step+1, "iteration": iteration, **metrics})
                print("step=%d t=%.4f iter=%d %s" % (step+1, time+dt, iteration, metrics), flush=True)
                if physics.accepted(metrics):
                    break
        if not physics.accepted(metrics):
            checkpoint(out/"checkpoint.pt", model, optimizer, c, state, step, replay, "residual_gate_failed")
            write_json(out/"status.json", {"status": "residual_gate_failed", "committed_steps": step,
                       "attempted_time": time+dt, "metrics": metrics})
            print("STOP: residual gate failed; checkpoint saved without advancing time.", flush=True)
            return 2, out
        if tr["replay_capacity"]:
            replay.append((state.detach().cpu(), time))
            replay = replay[-tr["replay_capacity"]:]
        previous_for_diagnostics = state
        state = velocity.detach()
        committed = step + 1
        save_field = committed % tr["save_every"] == 0 or committed == tr["steps"]
        record(out, physics, state, pressure, re, committed*dt, committed, save_field, previous=previous_for_diagnostics)
        # Save each accepted step, so failure never commits a rejected prediction.
        checkpoint(out/"checkpoint.pt", model, optimizer, c, state, committed, replay, "training")
    write_json(out/"status.json", {"status": "training_complete_not_physical_validation",
                                 "committed_steps": committed})
    checkpoint(out/"checkpoint.pt", model, optimizer, c, state, committed, replay, "training_complete")
    return 0, out



def main():
    code, out = train()
    if settings.PLOT_AFTER_TRAIN:
        from predict import plot_training_results
        plot_training_results(out)
    print("训练结果目录:", out, flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
