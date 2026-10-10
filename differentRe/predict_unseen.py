import csv
import json
import math
import inspect
import numpy as np
import params as settings


DEVICE = "cuda:0"
PREDICT_CHECKPOINT = None
PREDICT_OUTPUT = "runs/refined_unseen_corrected"
PREDICT_RE = [85.0, 105.0, 125.0, 145.0, 75.0, 155.0]
PREDICT_AMPLITUDE = 0.0015
PREDICT_TIMES = [10.0, 30.0, 50.0, 100.0]
PREDICT_LOG_EVERY = 20
PLOT_DISCARD_TIME = 50.0



VORTICITY_START = 1.0
VORTICITY_END = 100.0
VORTICITY_INTERVAL = 1.0
VORTICITY_FPS = 10
VORTICITY_LIMIT = None
VORTICITY_DPI = 100
MAKE_VORTICITY_GIF = True


REPLOT_DIRECTORY = None


CORRECTION_MAX_ITERATIONS = 1500
CORRECTION_LR = 0.0001
CORRECTION_CHECK_EVERY = 25



def correct_step(physics, model, optimizer, previous, reynolds, time, config, out, step):
    """Keep previous state fixed. Optimize only this step's ORIGINAL physics loss.

    Explicit enable_grad permits correction inside the outer no_grad rollout.
    Within a Re, accepted adapted weights/Adam moments carry forward to the next step.
    """
    import torch
    from model import append_csv
    previous = previous.detach()
    model.eval()
    with torch.no_grad():
        velocity, pressure = physics.predict(model, previous, reynolds, time)
        _, before = physics.residual(previous, velocity, pressure, reynolds)
    if physics.accepted(before):
        return velocity, pressure, before, before, 0, True, "direct_pass"

    print("Correct Re", float(reynolds[0]), "step", step, "initial residual:", before, flush=True)
    append_csv(out/"correction_iterations.csv", {"step": step, "time": time+config["physics"]["dt"],
               "updates": 0, "passed": False, "reason": "initial_gate_failed", **before})
    updates, reason = 0, "correction_budget_exhausted"
    model.train()
    with torch.enable_grad():
        for iteration in range(1, CORRECTION_MAX_ITERATIONS+1):
            optimizer.zero_grad(set_to_none=True)
            candidate, p_candidate = physics.predict(model, previous, reynolds, time)
            loss, _ = physics.residual(previous, candidate, p_candidate, reynolds)
            if not torch.isfinite(loss):
                reason = "nonfinite_loss"
                break
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["clip_grad"])
            if not torch.isfinite(norm):
                reason = "nonfinite_gradient"
                optimizer.zero_grad(set_to_none=True)
                break
            optimizer.step()
            updates += 1
            if iteration % CORRECTION_CHECK_EVERY == 0 or iteration == CORRECTION_MAX_ITERATIONS:
                # Check AFTER the parameter update, using the same acceptance function as before.
                model.eval()
                with torch.no_grad():
                    velocity, pressure = physics.predict(model, previous, reynolds, time)
                    _, metrics = physics.residual(previous, velocity, pressure, reynolds)
                passed = physics.accepted(metrics)
                append_csv(out/"correction_iterations.csv", {"step": step, "time": time+config["physics"]["dt"],
                           "updates": updates, "passed": passed,
                           "reason": "corrected_pass" if passed else "optimizing", **metrics})
                print("Correction step", step, "updates", updates, metrics, flush=True)
                if passed:
                    optimizer.zero_grad(set_to_none=True)
                    return velocity.detach(), pressure.detach(), metrics, before, updates, True, "corrected_pass"
                model.train()
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        velocity, pressure = physics.predict(model, previous, reynolds, time)
        _, metrics = physics.residual(previous, velocity, pressure, reynolds)
    # Numerical errors are reported as failure even if a cached/previous iterate had passed.
    append_csv(out/"correction_iterations.csv", {"step": step, "time": time+config["physics"]["dt"],
               "updates": updates, "passed": False, "reason": reason, **metrics})
    return velocity, pressure, metrics, before, updates, False, reason


def rollout():
    import torch
    from model import FNO, CylinderPhysics, get_device, validate, append_csv, write_json, record
    checkpoint_path = (settings.resolve_path(PREDICT_CHECKPOINT) if PREDICT_CHECKPOINT
                       else settings.latest_training_directory()/"checkpoint.pt")
    options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = True
    saved = torch.load(checkpoint_path, **options)
    if saved.get("version") == "time_query_fno_v1":
        raise ValueError("This version uses the original time-marching model and cannot load weights from the time-query model.")
    c = saved["config"]
    validate(c)
    dt = c["physics"]["dt"]
    if not PREDICT_TIMES or not PREDICT_RE or PREDICT_LOG_EVERY < 1:
        raise ValueError("Provide prediction Re, times and positive logging interval.")
    target_steps = {}
    for value in PREDICT_TIMES:
        time = float(value)
        if not math.isfinite(time) or time <= 0:
            raise ValueError("PREDICT_TIMES must be positive finite dimensionless times.")
        step = round(time/dt)
        if step < 1 or not math.isclose(step*dt, time, rel_tol=0, abs_tol=1e-7):
            raise ValueError("Target times must be integer multiples of dt stored in the checkpoint; this modification does not change the time step.")
        target_steps[step] = time
    reynolds_cases = [float(r) for r in PREDICT_RE]
    if any(not math.isfinite(r) or r <= 0 for r in reynolds_cases) or not math.isfinite(PREDICT_AMPLITUDE):
        raise ValueError("Invalid Re or initial amplitude.")
    trained_re = c["cases"]["re"]
    overlap = [r for r in reynolds_cases if any(math.isclose(r, x, abs_tol=1e-8) for x in trained_re)]
    if overlap:
        raise ValueError("This file is intended only for unseen Re cases. Remove the trained cases: " + str(overlap))
    if (not isinstance(CORRECTION_MAX_ITERATIONS, int) or CORRECTION_MAX_ITERATIONS < 1
            or not isinstance(CORRECTION_CHECK_EVERY, int) or CORRECTION_CHECK_EVERY < 1
            or not math.isfinite(CORRECTION_LR) or CORRECTION_LR <= 0):
        raise ValueError("Invalid local correction budget, check interval or learning rate.")
    from predict import vorticity_times
    sequence_times = vorticity_times(VORTICITY_START, VORTICITY_END, VORTICITY_INTERVAL)
    sequence_steps = set()
    for time in sequence_times:
        step = round(time/dt)
        if step < 1 or not math.isclose(step*dt, time, rel_tol=0, abs_tol=1e-7):
            raise ValueError("Vorticity output times must be positive and integer multiples of the checkpoint dt")
        sequence_steps.add(step)
    max_step = max(max(target_steps), max(sequence_steps))
    device = get_device(DEVICE)
    physics = CylinderPhysics(c, device)
    root = settings.fresh_output_directory(PREDICT_OUTPUT)
    write_json(root/"suite.json", {"checkpoint": str(checkpoint_path), "config": c,
               "prediction_Re": reynolds_cases, "amplitude": PREDICT_AMPLITUDE,
               "dimensionless_times": sorted(target_steps.values()), "requested_steps": max_step, "vorticity_times": sequence_times,
               "trained_steps": saved["committed_steps"], "checkpoint_status": saved.get("status"),
               "mode": "physics_corrected_new_case_solve",
               "correction_max_iterations": CORRECTION_MAX_ITERATIONS,
               "correction_lr": CORRECTION_LR, "correction_check_every": CORRECTION_CHECK_EVERY,
               "note": "Online optimization on the new Re using original physics loss and unchanged gates; NOT frozen generalization."})
    print("Checkpoint:", checkpoint_path, "| training Re:", c["cases"]["re"], flush=True)
    print("Prediction directory:", root, "| dt:", dt, "| total steps per case:", max_step, flush=True)
    all_passed = True
    with torch.no_grad():
        for index, reynolds in enumerate(reynolds_cases):
            training_re = c["cases"]["re"]
            kind = "seen" if reynolds in training_re else ("interpolation" if min(training_re) < reynolds < max(training_re) else "extrapolation")
            out = root/("%02d_Re_%g_%s" % (index, reynolds, kind))
            out.mkdir()
            # Start each Re from the ORIGINAL CPU checkpoint, never the preceding case's weights.
            model = FNO(**c["model"]).to(device)
            model.load_state_dict(saved["model"])
            model.requires_grad_(True)
            optimizer = torch.optim.Adam(model.parameters(), lr=CORRECTION_LR)
            re = torch.tensor([reynolds], dtype=torch.float32, device=device)
            state = physics.initial(torch.tensor([PREDICT_AMPLITUDE], dtype=torch.float32, device=device))
            accepted, failed_step, metrics = 0, None, {}
            corrected_steps, total_updates, direct_steps = 0, 0, 0
            stop_reason = "completed"
            for step in range(max_step):
                velocity, pressure, metrics, before, updates, passed, reason = correct_step(
                    physics, model, optimizer, state, re, step*dt, c, out, step+1)
                total_updates += updates
                if updates:
                    corrected_steps += 1
                elif passed:
                    direct_steps += 1
                append_csv(out/"residuals.csv", {"step": step+1, "time": (step+1)*dt,
                           "passed": passed, "correction_updates": updates, "reason": reason,
                           **{"before_"+key: value for key, value in before.items()}, **metrics})
                if not passed:
                    failed_step = step+1
                    stop_reason = reason
                    all_passed = False
                    print("Re", reynolds, "STOP at step", failed_step, reason, metrics, flush=True)
                    # 'state' remains the last accepted state; rejected fields are never committed.
                    break
                previous_for_diagnostics = state
                state = velocity.detach()
                accepted = step+1
                # Original record/physics functions unchanged. Dense force history; requested fields plus vorticity sequence times.
                record(out, physics, state, pressure, re, accepted*dt, accepted, (accepted in target_steps or accepted in sequence_steps), previous=previous_for_diagnostics)
                if accepted % PREDICT_LOG_EVERY == 0:
                    print("Re", reynolds, "t*=", accepted*dt, flush=True)
            reached = [t for s, t in sorted(target_steps.items()) if s <= accepted]
            missing = [t for s, t in sorted(target_steps.items()) if s > accepted]
            matched_pair = any(math.isclose(reynolds, r) and math.isclose(PREDICT_AMPLITUDE, a, abs_tol=1e-12)
                               for r, a in zip(training_re, c["cases"]["amplitude"]))
            report = {"Re": reynolds, "class": kind, "initial_amplitude": PREDICT_AMPLITUDE,
                      "seen_training_pair": matched_pair, "accepted_steps": accepted,
                      "failed_step": failed_step,
                      "vorticity_saved_times": [t for t in sequence_times if round(t/dt) <= accepted],
                      "vorticity_missing_times": [t for t in sequence_times if round(t/dt) > accepted],
                      "saved_times": reached, "missing_times": missing,
                      "time_extrapolation_requested": max_step > saved["committed_steps"],
                      "mode": "physics_corrected_new_case_solve",
                      "direct_pass_steps": direct_steps, "steps_with_correction": corrected_steps,
                      "total_correction_updates": total_updates, "stop_reason": stop_reason,
                      "status": "corrected_solve_gates_passed" if failed_step is None else "corrected_solve_stopped"}
            write_json(out/"status.json", {**report, "last_metrics": metrics})
            append_csv(root/"summary.csv", {**report, "saved_times": json.dumps(reached), "missing_times": json.dumps(missing)})
            if accepted:
                plot_results(out, case=0, discard_time=PLOT_DISCARD_TIME)
            else:
                print("No accepted states for Re", reynolds, "; inspect residuals.csv and status.json.", flush=True)
    return (0 if all_passed else 2), root




def plot_results(directory, case=0, discard_time=0.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = settings.resolve_path(directory)
    if not (path/"history.csv").exists():
        print("No accepted time steps are available for plotting. Please check status.json status. / loss.csv。")
        return False
    with (path/"history.csv").open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if int(r["case"]) == case]
    if not rows:
        print("No accepted data are available for the selected case")
        return False
    t = np.array([float(r["time"]) for r in rows])
    cd = np.array([float(r["Cd_penalty_indicator"]) for r in rows])
    cl = np.array([float(r["Cl_penalty_indicator"]) for r in rows])
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, cd); axes[0].set_ylabel("Cd penalty indicator")
    axes[1].plot(t, cl); axes[1].set_ylabel("Cl penalty indicator")
    axes[1].set_xlabel("t U_ref / D")
    fig.suptitle("Physics-corrected solve (not frozen inference)")
    fig.tight_layout(); fig.savefig(path/"force_indicators.png", dpi=160); plt.close(fig)
    report = {"note": "Spectral peak is a candidate only, not evidence of accurate sustained shedding."}
    tail = cl[t >= discard_time]
    times = t[t >= discard_time]
    if len(tail) >= 32 and np.std(tail) > 1e-8:
        uniform = np.allclose(np.diff(times), np.diff(times)[0], rtol=1e-4, atol=1e-7)
        if uniform:
            # Linear detrending and Hann window; D=U_ref=1 -> f equals St.
            detrended = tail - np.polyval(np.polyfit(times-times[0], tail, 1), times-times[0])
            spectrum = np.abs(np.fft.rfft(detrended*np.hanning(len(tail))))**2
            freq = np.fft.rfftfreq(len(tail), d=times[1]-times[0])
            peak = 1 + np.argmax(spectrum[1:])
            report.update(candidate_St=float(freq[peak]), cl_std=float(tail.std()),
                          frequency_bin=float(freq[1]), observed_candidate_cycles=float(freq[peak]*(times[-1]-times[0])))
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(freq[1:], spectrum[1:]); ax.set_xlabel("f D / U_ref"); ax.set_ylabel("Windowed power")
            fig.tight_layout(); fig.savefig(path/"spectrum.png", dpi=160); plt.close(fig)
        else:
            report["reason"] = "Nonuniform times; no FFT frequency estimate."
    else:
        report["reason"] = "Insufficient post-transient samples or negligible oscillation."
    (path/"spectral_diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fields = sorted(path.glob("field_*.npz"))
    for field_path in fields:
        with np.load(field_path, allow_pickle=False) as data:
            w = data["vorticity"][case, 0].copy()
            w[data["x"]**2 + data["y"]**2 <= 0.25] = np.nan
            limit = max(float(np.nanpercentile(np.abs(w), 99)), 1e-5)
            fig, ax = plt.subplots(figsize=(12, 5))
            mesh = ax.pcolormesh(data["x"], data["y"], w, cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto")
            ax.add_patch(plt.Circle((0, 0), 0.5, color="gray"))
            ax.set_aspect("equal"); ax.set_xlabel("x/D"); ax.set_ylabel("y/D")
            ax.set_title("Physics-corrected solve, Re=%.1f, t*=%.3f" % (data["re"][case], data["time"]))
            fig.colorbar(mesh, ax=ax, label="vorticity D/U_ref")
            fig.tight_layout(); fig.savefig(path/(field_path.stem+"_vorticity.png"), dpi=160); plt.close(fig)
    from predict import plot_force_diagnostics
    plot_force_diagnostics(path, case, discard_time, path)
    from predict import plot_vorticity_sequence, vorticity_times
    plot_vorticity_sequence(path, case, path,
        vorticity_times(VORTICITY_START, VORTICITY_END, VORTICITY_INTERVAL),
        VORTICITY_INTERVAL, VORTICITY_FPS, VORTICITY_LIMIT, VORTICITY_DPI, MAKE_VORTICITY_GIF)
    print("Plots saved to", path.resolve())
    return True




if __name__ == "__main__":
    if REPLOT_DIRECTORY is not None:
        directory = settings.resolve_path(REPLOT_DIRECTORY)
        cases = [directory] if (directory/"history.csv").exists() else sorted(
            p for p in directory.iterdir() if p.is_dir() and (p/"history.csv").exists())
        if not cases:
            raise SystemExit("No saved case results were found in this directory")
        success = [plot_results(p, case=0, discard_time=PLOT_DISCARD_TIME) for p in cases]
        raise SystemExit(0 if all(success) else 2)
    code, directory = rollout()
    print("Unseen-case results directory:", directory, flush=True)
    raise SystemExit(code)
