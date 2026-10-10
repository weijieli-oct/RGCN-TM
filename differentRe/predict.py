import csv
import json
import math
import numpy as np
import params as settings


TRAIN_RESULT_DIRECTORY = None
PLOT_RE = [80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0]
PLOT_TIMES = [10.0, 30.0, 50.0, 100.0]
PLOT_DISCARD_TIME = 50.0
FIGURE_DIRECTORY = None



VORTICITY_START = 1.0
VORTICITY_END = 100.0
VORTICITY_INTERVAL = 1.0
VORTICITY_FPS = 10
VORTICITY_LIMIT = None
VORTICITY_DPI = 100
MAKE_VORTICITY_GIF = True



PHASE_SEARCH_START = 50.0
PHASE_SEARCH_END = None
PHASE_MIN_LIFT_STD = 1e-6
PHASE_MAX_PERIOD_VARIATION = 0.15
PHASE_MAX_TIME_ERROR_FRACTION = 0.10
PHASE_MANUAL_CYCLES = {}


def plot_training_results(directory=None):
    root = (settings.resolve_path(directory) if directory is not None else
            settings.resolve_path(TRAIN_RESULT_DIRECTORY) if TRAIN_RESULT_DIRECTORY else settings.latest_training_directory())
    if not (root/"history.csv").exists():
        print("There are no existing training sequences that can be plotted:", root)
        return False
    if not PLOT_RE or not PLOT_TIMES or any(not math.isfinite(float(t)) or t < 0 for t in PLOT_TIMES):
        raise ValueError("Please fill in PLOT_RE and the valid dimensionless PLOT_TIMES")
    with (root/"history.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cases = {}
    for row in rows:
        cases[int(row["case"])] = float(row["Re"])
    output = settings.resolve_path(FIGURE_DIRECTORY) if FIGURE_DIRECTORY else root/"plots_existing"
    output.mkdir(parents=True, exist_ok=True)
    inventory = []
    for field_path in sorted(root.glob("field_*.npz")):
        with np.load(field_path, allow_pickle=False) as data:
            inventory.append((float(data["time"]), np.asarray(data["re"]).tolist()))
    sequence_times = vorticity_times(VORTICITY_START, VORTICITY_END, VORTICITY_INTERVAL)
    reports = []
    complete = True
    for reynolds in PLOT_RE:
        matches = [index for index, value in cases.items() if math.isclose(float(reynolds), value, rel_tol=0, abs_tol=1e-8)]
        if not matches:
            print("No training results", reynolds)
            reports.append({"Re": reynolds, "status": "case_missing", "missing_times": PLOT_TIMES})
            complete = False
            continue
        case = matches[0]
        found = [float(t) for t in PLOT_TIMES if any(math.isclose(float(t), saved_t, rel_tol=0, abs_tol=1e-6)
                 and case < len(rs) and math.isclose(float(reynolds), rs[case], rel_tol=0, abs_tol=1e-8)
                 for saved_t, rs in inventory)]
        missing = [float(t) for t in PLOT_TIMES if float(t) not in found]
        plot_results(root, case, PLOT_DISCARD_TIME, output/("Re_%g" % reynolds), found)
        sequence = plot_vorticity_sequence(root, case, output/("Re_%g" % reynolds),
            sequence_times, VORTICITY_INTERVAL, VORTICITY_FPS, VORTICITY_LIMIT,
            VORTICITY_DPI, MAKE_VORTICITY_GIF)
        if sequence["missing_times"]:
            complete = False
        reports.append({"vorticity_sequence": sequence, "Re": reynolds, "case": case, "saved_times": found, "missing_times": missing})
        if missing:
            complete = False
            print("Re", reynolds, "missing:", missing)
    (output/"plot_report.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    export_cycle_and_forces(root, output, cases)
    print("exist:", output)
    return complete


def plot_results(directory, case=0, discard_time=0.0, output=None, times=None):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = settings.resolve_path(directory)
    destination = path if output is None else settings.resolve_path(output)
    destination.mkdir(parents=True, exist_ok=True)
    if not (path/"history.csv").exists():
        print("No time steps have been accepted for plotting.")
        return False
    with (path/"history.csv").open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if int(r["case"]) == case]
    if not rows:
        print("The selected condition has no data")
        return False
    t = np.array([float(r["time"]) for r in rows])
    cd = np.array([float(r["Cd_penalty_indicator"]) for r in rows])
    cl = np.array([float(r["Cl_penalty_indicator"]) for r in rows])
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, cd); axes[0].set_ylabel("Cd penalty indicator")
    axes[1].plot(t, cl); axes[1].set_ylabel("Cl penalty indicator")
    axes[1].set_xlabel("t U_ref / D")
    fig.tight_layout(); fig.savefig(destination/"force_indicators.png", dpi=160); plt.close(fig)
    report = {"note": "Spectral peak is a candidate only, not evidence of accurate sustained shedding."}
    tail = cl[t >= discard_time]
    spectral_times = t[t >= discard_time]
    if len(tail) >= 32 and np.std(tail) > 1e-8:
        uniform = np.allclose(np.diff(spectral_times), np.diff(spectral_times)[0], rtol=1e-4, atol=1e-7)
        if uniform:
            # Linear detrending and Hann window; D=U_ref=1 -> f equals St.
            detrended = tail - np.polyval(np.polyfit(spectral_times-spectral_times[0], tail, 1), spectral_times-spectral_times[0])
            spectrum = np.abs(np.fft.rfft(detrended*np.hanning(len(tail))))**2
            freq = np.fft.rfftfreq(len(tail), d=spectral_times[1]-spectral_times[0])
            peak = 1 + np.argmax(spectrum[1:])
            report.update(candidate_St=float(freq[peak]), cl_std=float(tail.std()),
                          frequency_bin=float(freq[1]), observed_candidate_cycles=float(freq[peak]*(spectral_times[-1]-spectral_times[0])))
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(freq[1:], spectrum[1:]); ax.set_xlabel("f D / U_ref"); ax.set_ylabel("Windowed power")
            fig.tight_layout(); fig.savefig(destination/"spectrum.png", dpi=160); plt.close(fig)
        else:
            report["reason"] = "Nonuniform times; no FFT frequency estimate."
    else:
        report["reason"] = "Insufficient post-transient samples or negligible oscillation."
    (destination/"spectral_diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fields = sorted(path.glob("field_*.npz"))
    for field_path in fields:
        with np.load(field_path, allow_pickle=False) as data:
            if times is not None and not any(math.isclose(float(data["time"]), t, rel_tol=0, abs_tol=1e-6) for t in times):
                continue
            w = data["vorticity"][case, 0].copy()
            w[data["x"]**2 + data["y"]**2 <= 0.25] = np.nan
            limit = max(float(np.nanpercentile(np.abs(w), 99)), 1e-5)
            fig, ax = plt.subplots(figsize=(12, 5))
            mesh = ax.pcolormesh(data["x"], data["y"], w, cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto")
            ax.add_patch(plt.Circle((0, 0), 0.5, color="gray"))
            ax.set_aspect("equal"); ax.set_xlabel("x/D"); ax.set_ylabel("y/D")
            ax.set_title("Self-generated vorticity, Re=%.1f, t=%.3f" % (data["re"][case], data["time"]))
            fig.colorbar(mesh, ax=ax, label="vorticity D/U_ref")
            fig.tight_layout(); fig.savefig(destination/(field_path.stem+"_vorticity.png"), dpi=160); plt.close(fig)
    plot_force_diagnostics(path, case, discard_time, destination)
    print("Plots saved to", destination.resolve())
    return True


def plot_force_diagnostics(directory, case, discard_time, destination):
    """Plot new stress estimates without requiring torch or changing either entry's settings."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory, destination = settings.resolve_path(directory), settings.resolve_path(destination)
    source = directory/"force_history.csv"
    if not source.exists():
        return  # Old results remain usable; no synthetic stress coefficients are invented.
    with source.open(encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if int(row["case"]) == case]
    if not rows:
        return
    t = np.array([float(row["time"]) for row in rows], dtype=np.float64)
    array = lambda key: np.array([float(row[key]) for row in rows], dtype=np.float64)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, prefix in zip(axes, ("Cd", "Cl")):
        for suffix in ("total", "pressure", "viscous"):
            ax.plot(t, array(prefix+"_"+suffix), label=suffix)
        ax.set_ylabel(prefix); ax.legend()
    axes[-1].set_xlabel("t U_ref/D")
    fig.suptitle("Fluid-side reconstructed surface stress estimates (not calibrated)")
    fig.tight_layout(); fig.savefig(destination/"force_coefficients.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, prefix in zip(axes, ("Cd", "Cl")):
        for suffix in ("total", "shifted_estimate", "penalty_trapezoid", "penalty_inertia_indicator"):
            ax.plot(t, array(prefix+"_"+suffix), label=suffix)
        ax.set_ylabel(prefix); ax.legend(fontsize=8)
    axes[-1].set_xlabel("t U_ref/D")
    fig.tight_layout(); fig.savefig(destination/"force_method_comparison.png", dpi=160); plt.close(fig)
    mask = t >= discard_time
    times = t[mask]
    report = {"source": "fluid_side_reconstructed_surface_stress", "discard_time": discard_time,
              "note": "Window statistics and sensitivity only. No reference-error or exact-force claim."}
    if len(times) < 2 or not np.all(np.diff(times) > 0):
        report["reason"] = "Insufficient or non-increasing samples after selected start time."
    else:
        report.update(start_time=float(times[0]), end_time=float(times[-1]), samples=len(times))
        duration = times[-1]-times[0]
        for key in ("Cd_total", "Cl_total", "Cd_pressure", "Cl_pressure", "Cd_viscous", "Cl_viscous",
                    "Cp_front_minus_back", "Cd_ring_sensitivity", "Cl_ring_sensitivity",
                    "Cd_quadrature_difference", "Cl_quadrature_difference"):
            values = array(key)[mask]
            if not np.isfinite(values).all():
                report[key] = {"reason": "nonfinite samples"}
                continue
            mean = float(np.trapz(values, times)/duration)
            report[key] = {"mean": mean, "rms": float(np.sqrt(np.trapz(values**2, times)/duration)),
                "fluctuation_rms": float(np.sqrt(np.trapz((values-mean)**2, times)/duration)),
                "min": float(values.min()), "max": float(values.max()), "peak_to_peak": float(np.ptp(values))}
        cl = array("Cl_total")[mask]
        uniform = np.allclose(np.diff(times), np.diff(times)[0], rtol=1e-4, atol=1e-8)
        if len(times) >= 32 and uniform and np.isfinite(cl).all() and np.std(cl) > 1e-8:
            detrended = cl-np.polyval(np.polyfit(times-times[0], cl, 1), times-times[0])
            spectrum = np.abs(np.fft.rfft(detrended*np.hanning(len(cl))))**2
            frequency = np.fft.rfftfreq(len(cl), d=times[1]-times[0])
            peak = 1+int(np.argmax(spectrum[1:]))
            report["frequency"] = {"candidate_St": float(frequency[peak]), "bin_width": float(frequency[1]),
                "candidate_cycles": float(frequency[peak]*duration), "method": "Hann FFT after linear detrending"}
            centered = cl-float(np.mean(cl))
            up = np.flatnonzero((centered[:-1] <= 0)&(centered[1:] > 0))
            crossing_times = times[up]-centered[up]*(times[up+1]-times[up])/(centered[up+1]-centered[up])
            if len(crossing_times) >= 4:
                periods = np.diff(crossing_times)
                report["frequency"].update(zero_crossing_St=float(1/np.median(periods)),
                    crossing_period_variation=float(np.std(periods)/np.mean(periods)))
            halves = []
            for indices in np.array_split(np.arange(len(cl)), 2):
                values = cl[indices]
                halves.append({"start": float(times[indices[0]]), "end": float(times[indices[-1]]),
                               "Cl_mean": float(np.mean(values)), "Cl_fluctuation_rms": float(np.std(values))})
            report["half_window_comparison"] = halves
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(frequency[1:], spectrum[1:]); ax.set_xlabel("St = f D/U_ref"); ax.set_ylabel("Windowed power")
            fig.tight_layout(); fig.savefig(destination/"lift_stress_spectrum.png", dpi=160); plt.close(fig)
        else:
            report["frequency"] = {"reason": "Need at least32 uniform finite samples and nonnegligible oscillation."}
    (destination/"force_statistics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")





def vorticity_times(start, end, interval):
    """Uniform requested dimensionless times, inclusive endpoint when aligned."""
    if not all(math.isfinite(float(v)) for v in (start, end, interval)):
        raise ValueError("The moment setting of the vorticity must be a finite number.")
    if start < 0 or end < start or interval <= 0:
        raise ValueError("Must 0 <= START <= END，INTERVAL > 0。")
    count = (end-start)/interval
    if not math.isclose(count, round(count), rel_tol=0, abs_tol=1e-6):
        raise ValueError("END-START Must be an integer multiple of INTERVAL")
    return [float(start+i*interval) for i in range(round(count)+1)]


def plot_vorticity_sequence(directory, case, output, requested_times, interval=1.0,
                            fps=10, limit=None, dpi=100, make_gif=True):
    """Only real saved snapshots; fixed color scale, no temporal interpolation.

    Each GIF hold duration reflects the gap to the next saved time. The last
    frame holds for one requested interval. Missing times are explicitly reported.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    if not math.isfinite(fps) or fps <= 0 or fps > 100 or dpi <= 0:
        raise ValueError("Must 0 < FPS <= 100、DPI > 0。")
    if limit is not None and (not math.isfinite(limit) or limit <= 0):
        raise ValueError("VORTICITY_LIMIT is None")
    source = settings.resolve_path(directory)
    parent = settings.resolve_path(output)/"vorticity_sequence"
    parent.mkdir(parents=True, exist_ok=True)
    # Each rendering has its own folder: stale frames/GIF from older runs cannot mix in.
    from datetime import datetime
    destination = parent/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination.mkdir()
    chosen = {}
    expected_re = None
    for field in sorted(source.glob("field_*.npz")):
        with np.load(field, allow_pickle=False) as data:
            time = float(data["time"])
            if case >= len(data["re"]):
                continue
            matches = [t for t in requested_times if math.isclose(t, time, rel_tol=0, abs_tol=1e-6)]
            if not matches:
                continue
            re = float(data["re"][case])
            if expected_re is not None and not math.isclose(expected_re, re, abs_tol=1e-8):
                raise ValueError("Snapshots of the same case with different Re values cannot be merged as they represent different animations.")
            expected_re = re
            if matches[0] in chosen:
                raise ValueError("If duplicate snapshots are found, please first check the result directory.。")
            chosen[matches[0]] = (time, field)
    selected = sorted(chosen.values(), key=lambda pair: pair[0])
    missing = [t for t in requested_times if t not in chosen]
    report = {"requested_times": requested_times, "saved_times": [t for t, _ in selected],
              "missing_times": missing, "Re": expected_re, "output": str(destination),
              "fps_at_requested_interval": fps, "requested_interval": interval,
              "complete": not missing, "animation": None,
              "note": "Saved fields only; missing times are not synthesized. GIF duration follows saved time gaps."}
    def read_field(field):
        with np.load(field, allow_pickle=False) as data:
            x, y = data["x"].copy(), data["y"].copy()
            w = data["vorticity"][case, 0].astype(float)
        w[x*x+y*y <= 0.25] = np.nan
        if not np.isfinite(w).any() or np.isinf(w).any():
            raise ValueError("The snapshot does not have a valid vorticity or contains infinite values: " + str(field))
        return x, y, w
    if selected:
        color_limit = float(limit) if limit is not None else max(
            1e-5, max(float(np.nanmax(np.abs(read_field(f)[2]))) for _, f in selected))
        report["color_limit"] = color_limit
        x, y, w = read_field(selected[0][1])
        fig, ax = plt.subplots(figsize=(10, 4), dpi=dpi)
        mesh = ax.pcolormesh(x, y, w, cmap="RdBu_r", vmin=-color_limit,
                             vmax=color_limit, shading="auto")
        ax.add_patch(plt.Circle((0, 0), 0.5, color="gray"))
        ax.set_aspect("equal"); ax.set_xlabel("x/D"); ax.set_ylabel("y/D")
        title = ax.set_title("")
        fig.colorbar(mesh, ax=ax, label="vorticity D/U_ref")
        fig.tight_layout()
        frames, durations = [], []
        try:
            for index, (time, field) in enumerate(selected):
                next_x, next_y, w = read_field(field)
                if not np.array_equal(x, next_x) or not np.array_equal(y, next_y):
                    raise ValueError("The grid of the animation frames is not consistent.")
                mesh.set_array(np.ma.masked_invalid(w).ravel())
                title.set_text("Vorticity, Re=%g, t*=%.6g" % (expected_re, time))
                png = destination/("vorticity_%04d_t_%09.4f.png" % (index, time))
                fig.savefig(png, dpi=dpi)
                if make_gif:
                    with Image.open(png) as im:
                        adaptive = getattr(Image, "Palette", Image).ADAPTIVE
                        frames.append(im.convert("RGB").convert("P", palette=adaptive))
                    gap = selected[index+1][0]-time if index+1 < len(selected) else interval
                    durations.append(max(10, int(round(100*gap/interval/fps))*10))
                if (index+1) % 20 == 0 or index+1 == len(selected):
                    print("Vorticity frames:", index+1, "/", len(selected), "Re=", expected_re, flush=True)
            if make_gif and len(frames) >= 2:
                gif = destination/("vorticity_partial.gif" if missing else "vorticity_animation.gif")
                frames[0].save(gif, save_all=True, append_images=frames[1:],
                               duration=durations, loop=0, optimize=False, disposal=2)
                report["animation"] = str(gif)
                report["playback_seconds"] = sum(durations)/1000
            elif make_gif:
                report["animation_reason"] = "At least two saved times are required."
        finally:
            plt.close(fig)
            for frame in frames:
                frame.close()
    if missing:
        print("vorticity sequence is missing", len(missing), "time points；only the existing snapshots are plotted", destination/"sequence_report.json", flush=True)
    (destination/"sequence_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Vorticity sequence saved to", destination, flush=True)
    return report


def _force_rows(root, filename, case=None):
    path = root/filename
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig") as stream:
        return [row for row in csv.DictReader(stream)
                if (case is None or int(row["case"]) == case)
                and any(math.isclose(float(row["Re"]), re, abs_tol=1e-8) for re in PLOT_RE)]


def _identify_cycle(rows, key):
    """Candidate cycle from upward crossings of the late-window mean lift."""
    pairs = sorted((float(r["time"]), float(r[key])) for r in rows
                   if float(r["time"]) >= PHASE_SEARCH_START
                   and (PHASE_SEARCH_END is None or float(r["time"]) <= PHASE_SEARCH_END))
    if len(pairs) < 32:
        return None, {"reason": "insufficient_lift_samples"}
    t, cl = np.array(pairs).T
    if not np.isfinite(t).all() or not np.isfinite(cl).all() or np.any(np.diff(t) <= 0):
        return None, {"reason": "invalid_or_duplicate_lift_samples"}
    if not np.allclose(np.diff(t), np.diff(t)[0], rtol=1e-4, atol=1e-7):
        return None, {"reason": "nonuniform_lift_samples"}
    mean = float(np.mean(cl)); centered = cl-mean
    if np.std(centered) < PHASE_MIN_LIFT_STD:
        return None, {"reason": "negligible_lift_oscillation"}
    indices = np.flatnonzero((centered[:-1] <= 0) & (centered[1:] > 0))
    crosses = t[indices]-centered[indices]*np.diff(t)[indices]/(centered[indices+1]-centered[indices])
    if len(crosses) < 4:
        return None, {"reason": "need_at_least_three_observed_cycles"}
    periods = np.diff(crosses)[-3:]
    variability = float(np.std(periods)/np.mean(periods))
    info = {"method": "last upward mean-lift crossing pair", "lift_mean": mean,
            "last_three_periods": periods.tolist(), "period_variation": variability,
            "note": "Candidate cycle only; not independent physical validation."}
    if variability > PHASE_MAX_PERIOD_VARIATION:
        return None, {**info, "reason": "irregular_recent_periods"}
    period = float(periods[-1])
    if period/np.diff(t)[0] < 16:
        return None, {**info, "reason": "too_few_lift_samples_per_cycle"}
    return (float(crosses[-2]), period), info


def _select_phase_fields(fields, start, period):
    targets = [start+fraction*period for fraction in (0., .25, .5, .75)]
    if len(fields) < 4:
        return [], "fewer_than_four_snapshots"
    selected = []
    for fraction, target in zip((0., .25, .5, .75), targets):
        time, path = min(fields, key=lambda pair: abs(pair[0]-target))
        selected.append({"target_phase": fraction, "target_time": target, "actual_time": time,
                         "actual_phase": (time-start)/period, "phase_error": (time-target)/period,
                         "snapshot": str(path)})
    if len({r["snapshot"] for r in selected}) != 4:
        return selected, "snapshots_not_distinct"
    if any(abs(r["phase_error"]) > PHASE_MAX_TIME_ERROR_FRACTION+1e-10 for r in selected):
        return selected, "snapshots_too_far_from_target_phases"
    return selected, None


def export_cycle_and_forces(root, output, cases):
    """Additional postprocessing only. Never modifies histories, models or existing plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
    from datetime import datetime
    if not (math.isfinite(PHASE_SEARCH_START) and PHASE_SEARCH_START >= 0
            and (PHASE_SEARCH_END is None or math.isfinite(PHASE_SEARCH_END) and PHASE_SEARCH_END > PHASE_SEARCH_START)
            and 0 <= PHASE_MAX_PERIOD_VARIATION < 1 and 0 < PHASE_MAX_TIME_ERROR_FRACTION < .125
            and math.isfinite(PHASE_MIN_LIFT_STD) and PHASE_MIN_LIFT_STD > 0):
        raise ValueError("Invalid cycle search window or threshold parameters")
    root, output = settings.resolve_path(root), settings.resolve_path(output)
    destination = output/"cycle_and_forces"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination.mkdir(parents=True)
    workbook = Workbook(); notes = workbook.active; notes.title = "说明"
    notes.append(["Item", "Description"])
    for row in [("Source directory", str(root)), ("Cases", str(PLOT_RE)),
                ("Time", "t*=t U_ref/D; all saved time steps are exported, not limited to the four phase instants"),
                ("Total coefficients",
                 "Cd_total/Cl_total: reconstructed estimates of the fluid-side surface stresses from the original model"),
                ("Components",
                 "Cd_pressure/Cl_pressure and Cd_viscous/Cl_viscous: original pressure and viscous components"),
                ("Penalty terms",
                 "Fields such as penalty retain their original names and must not be interpreted as calibrated surface-stress coefficients"),
                ("Data processing",
                 "Original CSV values are transferred without smoothing or recalculation; NaN/Inf text values are preserved and are not replaced by zero"),
                ("Cycle",
                 "Cl_total is preferentially used for automatic cycle identification; if unavailable in legacy results, Cl_penalty_indicator is used and the source is recorded"),
                ("Phase",
                 "Zero phase is defined by the upward crossing of the late-time mean lift; for a manually specified cycle, the prescribed starting time is used"),
                ("Snapshots",
                 "Nearest available snapshots are used with a common color scale and without temporal interpolation; actual_phase denotes the actual phase and may differ from the target"),
                ("Maximum error fraction of T", PHASE_MAX_TIME_ERROR_FRACTION)]:
        notes.append(row)
    for filename, title in (("force_history.csv", "surface_force_history"), ("history.csv", "penalty_history")):
        rows = _force_rows(root, filename)
        if not rows:
            notes.append([filename, "No data available for the selected cases"]); continue
        sheet = workbook.create_sheet(title)
        headers = list(rows[0]); sheet.append(headers)
        for row in sorted(rows, key=lambda r: (float(r["Re"]), int(r["case"]), float(r["time"]))):
            values = []
            for key in headers:
                value = row[key]
                try:
                    number = float(value)
                    value = int(number) if key in ("case", "step") and math.isfinite(number) else number if math.isfinite(number) else value
                except (ValueError, TypeError):
                    pass
                values.append(value)
            sheet.append(values)
    phase_sheet = workbook.create_sheet("cycle_phases")
    phase_headers = ["case", "Re", "lift_source", "cycle_start", "period", "target_phase", "target_time",
                     "actual_time", "actual_phase", "phase_error", "snapshot", "status"]
    phase_sheet.append(phase_headers)
    reports = []
    for case, re in cases.items():
        if not any(math.isclose(re, r, abs_tol=1e-8) for r in PLOT_RE): continue
        rows = _force_rows(root, "force_history.csv", case); key = "Cl_total"
        if not rows or key not in rows[0]:
            rows = _force_rows(root, "history.csv", case); key = "Cl_penalty_indicator"
        manual = next((value for r, value in PHASE_MANUAL_CYCLES.items() if math.isclose(float(r), re, abs_tol=1e-8)), None)
        if manual is not None:
            if len(manual) != 2 or not all(math.isfinite(float(v)) for v in manual) or manual[0] < 0 or manual[1] <= 0:
                raise ValueError("Manual cycle format must be Re: (non-negative t0, positive T).")
            cycle, detail = tuple(map(float, manual)), {"method": "user-specified start and period"}
        else:
            cycle, detail = _identify_cycle(rows, key)
        report = {"case": case, "Re": re, "lift_source": key, **detail}
        selection, reason = [], detail.get("reason")
        if cycle:
            start, period = cycle
            fields = []
            for path in root.glob("field_*.npz"):
                with np.load(path, allow_pickle=False) as data:
                    if case < len(data["re"]) and math.isclose(float(data["re"][case]), re, abs_tol=1e-8):
                        fields.append((float(data["time"]), path))
            fields.sort()
            selection, reason = _select_phase_fields(fields, start, period)
            report.update(cycle_start=start, period=period, phases=selection)
        report["status"] = reason or "four_nearest_phase_snapshots"
        reports.append(report)
        for item in selection or [{}]:
            values = {**report, **item}
            phase_sheet.append([values.get(k) for k in phase_headers])
        if reason:
            print("Re", re, "four-phase plot was not generated:", reason, flush=True)
            continue
        frames = []
        for item in selection:
            with np.load(item["snapshot"], allow_pickle=False) as data:
                x, y = data["x"].copy(), data["y"].copy()
                w = data["vorticity"][case, 0].astype(float)
            solid = x*x+y*y <= .25
            if not np.isfinite(w[~solid]).all():
                raise ValueError("Non-finite vorticity values detected in the fluid region: "+item["snapshot"])
            w[solid] = np.nan; frames.append((x, y, w))
        limit = max(1e-5, max(float(np.nanmax(np.abs(w))) for x,y,w in frames))
        fig, axes = plt.subplots(2, 2, figsize=(15, 8), constrained_layout=True)
        for index, (ax, item, (x,y,w)) in enumerate(zip(axes.flat, selection, frames)):
            mesh = ax.pcolormesh(x,y,w,cmap="RdBu_r",vmin=-limit,vmax=limit,shading="auto")
            ax.add_patch(plt.Circle((0,0),.5,color="gray"))
            ax.set_aspect("equal"); ax.set_xlabel("x/D"); ax.set_ylabel("y/D")
            ax.set_title(f"({chr(97+index)}) target {item['target_phase']:.2f}T; actual {item['actual_phase']:.3f}T\nt*={item['actual_time']:.4f}")
        fig.colorbar(mesh,ax=list(axes.flat),label="vorticity D/U_ref",shrink=.85)
        fig.suptitle(f"Re={re:g}, candidate T*={period:.4f}; nearest saved snapshots")
        stem = destination/f"Re_{re:g}_case_{case}_four_phases"
        fig.savefig(str(stem)+".png",dpi=180); fig.savefig(str(stem)+".pdf"); plt.close(fig)
    for sheet in workbook:
        sheet.freeze_panes="A2"; sheet.auto_filter.ref=sheet.dimensions
        for cell in sheet[1]:
            cell.font=Font(bold=True,color="FFFFFF"); cell.fill=PatternFill("solid",fgColor="24476B")
        for index,cells in enumerate(sheet.columns,1):
            sheet.column_dimensions[get_column_letter(index)].width=min(55,max(14,max(len(str(c.value or "")) for c in cells)+2))
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                if isinstance(cell.value,float): cell.number_format="0.00000000"
        if sheet.title=="说明":
            sheet.column_dimensions['B'].width=90
            for row in sheet.iter_rows(min_row=2):
                row[1].alignment=Alignment(wrap_text=True,vertical="top")
                sheet.row_dimensions[row[0].row].height=32
    workbook.save(destination/"lift_drag_coefficients.xlsx")
    (destination/"cycle_report.json").write_text(json.dumps(reports,indent=2,ensure_ascii=False),encoding="utf-8")
    print("Four-phase plots and lift/drag coefficient Excel file:",destination,flush=True)
    return destination


if __name__ == "__main__":
    raise SystemExit(0 if plot_training_results() else 2)
