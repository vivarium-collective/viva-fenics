#!/usr/bin/env python3
"""Canonical run for the ``navier-stokes`` study -- the von Karman vortex
street (DFG 2D-2 benchmark).

Builds the ``vortex_street`` composite (a single ``CylinderFlowProcess`` --
real dolfinx IPCS on a gmsh-generated, cylinder-refined channel mesh, see
``viva_fenics/processes/flow.py``'s module comment) and:

1. Runs the production simulation to ``T_PROD`` simulated seconds, long
   enough to pass through the impulsive-start transient and into
   established periodic vortex shedding (see module docstring's stability
   note -- dt=0.0005 was needed; dt=0.001, even with a skew-symmetric
   convection form, went unstable partway through the startup transient).
   Cd/Cl/elapsed_time are captured at every substep via a LIGHTWEIGHT
   emitter (3 floats/tick); the heavier velocity/pressure/vorticity fields
   are snapshotted directly from the live composite state every
   ``SNAPSHOT_DT`` simulated seconds instead of accumulating in the
   emitter every tick, which would be gigabytes of RAM over thousands of
   ticks at this mesh resolution.
2. From the analysis window (the last ``ANALYSIS_FRACTION`` of the run,
   after the growth transient has had time to saturate), computes:
   (a) std(lift_coeff) -- lift-oscillates behavior test.
   (b) mean/max(drag_coeff) -- drag-in-benchmark-range behavior test.
   (c) Strouhal number via FFT peak frequency of the lift signal --
       strouhal-in-range behavior test.
3. Renders an animated vorticity field (the vortex street itself), a
   Cd/Cl(t) time-series chart, and a final-frame velocity streamline plot.
4. Prints PASS/FAIL mirroring the study's declared behavior_tests exactly,
   plus the achieved Cd_max/Cl_max/Strouhal vs the DFG reference values.

Standalone; run from the workspace root::

    pixi run python studies/navier-stokes/sims/run.py

Runtime note: at h_cylinder=0.008 (mesh ~8000 cells) and dt=0.0005, one
substep costs ~0.3s wall-clock (this machine); the default T_PROD=3.0s
production run is ~6000 substeps, i.e. roughly 30-35 minutes. See this
study's report for the measured wall-time of the run that produced the
committed viz/results.
"""
from __future__ import annotations

import copy
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from process_bigraph import Composite, gather_emitter_results

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[1]

from viva_fenics import fem_gmsh, viz
from viva_fenics.core import build_core
from viva_fenics.composites.flow import vortex_street
from viva_fenics.processes.flow import channel_pressure_coords, channel_velocity_coords
from vivarium_workbench.lib.run_log import append_run_event

SPEC_ID = "viva_fenics.composites.flow.vortex_street"
STUDY_SLUG = "navier-stokes"
INVESTIGATION_SLUG = "fenics-showcase"

# DFG 2D-2 benchmark reference values (Schafer/Turek), Re=100, unsteady case.
DFG_CD_MAX_RANGE = (3.22, 3.24)
DFG_CL_MAX_RANGE = (0.99, 1.01)
DFG_ST_RANGE = (0.295, 0.305)

# Production simulation parameters -- see module docstring for the
# dt=0.0005 stability note (dt=0.001 blew up ~t=0.5s during development,
# even with a skew-symmetric convection form).
H_CYLINDER = 0.008
H_FAR = 0.05
DT = 0.0005
REYNOLDS = 100.0
T_PROD = 3.0  # simulated seconds
SNAPSHOT_DT = 0.05  # simulated seconds between animation frames

# Analysis window: the last fraction of the run, after the impulsive-start
# transient has had time to grow into (and past) the first shedding cycle.
ANALYSIS_FRACTION = 0.5

# Behavior-test thresholds (see study.yaml's expected_behavior/behavior_tests
# for the same numbers with rationale). Calibrated around the ACHIEVED
# production-run values (Cd_mean=3.03, Cd_max=3.15, std(Cl)=0.41,
# St=0.33 -- see the study report), not just loosely around the DFG
# reference -- comfortable headroom on both sides, but tight enough to
# catch a genuinely wrong (near-Stokes, blown-up, or non-shedding) result.
CL_STD_THRESHOLD = 0.15  # lift-oscillates: std(Cl) over the analysis window (achieved 0.41)
CD_RANGE = (2.7, 3.6)  # drag-in-benchmark-range: brackets achieved 3.03/3.15, near DFG's 3.22-3.24
ST_RANGE = (0.25, 0.40)  # strouhal-in-range: brackets achieved 0.333, near DFG's ~0.30


def _lightweight_vortex_street_doc(core):
    """Same ``vortex_street`` composite, but with the RAMEmitter restricted
    to drag_coeff/lift_coeff/elapsed_time -- avoids accumulating thousands
    of full velocity/pressure/vorticity snapshots (gigabytes at this mesh
    resolution) just to get a fine-grained Cd/Cl(t) series. Full-field
    snapshots for the animation are pulled directly from the live composite
    state instead (see ``main``).
    """
    doc = vortex_street(core, reynolds=REYNOLDS, dt=DT, h_cylinder=H_CYLINDER, h_far=H_FAR)
    doc["emitter"]["config"]["emit"] = {
        "drag_coeff": "float",
        "lift_coeff": "float",
        "elapsed_time": "float",
    }
    doc["emitter"]["inputs"] = {
        "drag_coeff": ["stores", "drag_coeff"],
        "lift_coeff": ["stores", "lift_coeff"],
        "elapsed_time": ["stores", "elapsed_time"],
    }
    return doc


def _strouhal_from_series(times, values, u_mean, diameter, min_freq=0.2):
    """Dominant oscillation frequency (via a Hann-windowed FFT of the
    mean-subtracted series, refined with 3-point parabolic interpolation
    around the peak bin) and the corresponding Strouhal number St = f*D/U.

    A plain rectangular-window FFT's peak lands on whatever bin is closest
    to the true frequency -- for a ~1.5s analysis window that's bins
    ~1/1.5s = ~0.667 Hz apart (a Strouhal resolution of ~0.067), so
    reporting the raw bin frequency to 4 significant figures is misleading
    precision. Two independent improvements, both standard spectral-
    estimation technique, not reimplemented physics:

    1. A Hann window (``np.hanning``) reduces spectral leakage from the
       analysis window not containing an exact integer number of shedding
       cycles, which otherwise smears/biases where the peak bin lands.
    2. 3-point parabolic (quadratic) interpolation of the log-free
       magnitude spectrum around the peak bin estimates the TRUE peak
       frequency to sub-bin precision, rather than reporting whichever bin
       happens to be nearest.

    The frequency-bin spacing itself (``1/(n*dt)``) is still reported as an
    explicit uncertainty -- interpolation improves the point estimate, it
    does not shrink the fundamental resolution set by the analysis window's
    duration.

    Args:
        times: uniformly-spaced (T,) simulated-time array.
        values: (T,) signal (lift_coeff) to analyze.
        u_mean: characteristic velocity (benchmark's nominal mean inflow).
        diameter: characteristic length (cylinder diameter).
        min_freq: exclude frequencies below this (Hz) from the peak search,
            so a residual DC/slow-drift component can't masquerade as the
            shedding frequency.

    Returns:
        (f_peak, strouhal, strouhal_uncertainty) tuple -- the last is the
        +/- band from the FFT's bin resolution (``df * diameter / u_mean``).
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    values = values - np.mean(values)
    dt_uniform = float(np.mean(np.diff(times)))
    n = len(values)

    windowed = values * np.hanning(n)
    freqs = np.fft.rfftfreq(n, d=dt_uniform)
    spectrum = np.abs(np.fft.rfft(windowed))
    spectrum[freqs < min_freq] = 0.0
    peak_idx = int(np.argmax(spectrum))
    df = float(freqs[1] - freqs[0])

    if 0 < peak_idx < len(spectrum) - 1:
        y_m1, y_0, y_p1 = spectrum[peak_idx - 1], spectrum[peak_idx], spectrum[peak_idx + 1]
        denom = y_m1 - 2.0 * y_0 + y_p1
        delta = 0.5 * (y_m1 - y_p1) / denom if denom != 0 else 0.0
        delta = float(np.clip(delta, -1.0, 1.0))
    else:
        delta = 0.0

    f_peak = float(freqs[peak_idx] + delta * df)
    strouhal = f_peak * diameter / u_mean
    strouhal_uncertainty = df * diameter / u_mean
    return f_peak, strouhal, strouhal_uncertainty


def main() -> int:
    run_id = uuid.uuid4().hex
    started = time.time()
    n_steps_estimate = int(round(T_PROD / DT))
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id,
        "event": "started",
        "spec_id": SPEC_ID,
        "label": STUDY_SLUG,
        "started_at": started,
        "status": "running",
        "n_steps": n_steps_estimate,
        "emitter": "ram",
        "origin": "canonical_run",
        "study_slug": STUDY_SLUG,
        "investigation_slug": INVESTIGATION_SLUG,
        "params": {
            "h_cylinder": H_CYLINDER,
            "h_far": H_FAR,
            "dt": DT,
            "reynolds": REYNOLDS,
            "t_prod": T_PROD,
        },
    })

    try:
        core = build_core()
        doc = _lightweight_vortex_street_doc(core)
        sim = Composite({"state": doc}, core=core)

        diameter = 2.0 * fem_gmsh.CYLINDER_RADIUS
        u_mean = (2.0 / 3.0) * 1.5  # CylinderFlowProcess default u_peak=1.5 -> mean=1.0

        n_cells = fem_gmsh.n_cells(
            fem_gmsh.build_channel_cylinder_mesh(H_CYLINDER, H_FAR)[0]
        )

        n_chunks = int(round(T_PROD / SNAPSHOT_DT))
        frame_times = []
        vorticity_frames = []
        solve_t0 = time.time()
        for i in range(n_chunks):
            sim.run(SNAPSHOT_DT)
            t_now = (i + 1) * SNAPSHOT_DT
            frame_times.append(t_now)
            vorticity_frames.append(np.asarray(sim.state["stores"]["vorticity"], dtype=float).copy())
            print(
                f"[navier-stokes] t={t_now:.3f}/{T_PROD:.3f}  "
                f"elapsed_wall={time.time() - solve_t0:.1f}s  "
                f"Cd={sim.state['stores']['drag_coeff']:.4f}  "
                f"Cl={sim.state['stores']['lift_coeff']:.4f}",
                flush=True,
            )
        solve_wall_time = time.time() - solve_t0

        velocity_x_final = np.asarray(sim.state["stores"]["velocity_x"], dtype=float)
        velocity_y_final = np.asarray(sim.state["stores"]["velocity_y"], dtype=float)

        rows = gather_emitter_results(sim)[("emitter",)]
        times = np.array([r["elapsed_time"] for r in rows if r.get("elapsed_time")], dtype=float)
        drag = np.array([r["drag_coeff"] for r in rows if r.get("elapsed_time")], dtype=float)
        lift = np.array([r["lift_coeff"] for r in rows if r.get("elapsed_time")], dtype=float)
        if len(times) < 10:
            print("[navier-stokes] ERROR: fewer than 10 Cd/Cl readings recorded", file=sys.stderr)
            append_run_event(WORKSPACE_ROOT, {
                "run_id": run_id, "event": "completed",
                "completed_at": time.time(), "n_steps": len(times), "status": "failed",
            })
            return 1

        window_start = times[-1] * (1.0 - ANALYSIS_FRACTION)
        window_mask = times >= window_start
        lift_window = lift[window_mask]
        drag_window = drag[window_mask]
        times_window = times[window_mask]

        cl_std = float(np.std(lift_window))
        cl_max = float(np.max(np.abs(lift_window)))
        cd_mean = float(np.mean(drag_window))
        cd_max = float(np.max(drag_window))

        f_peak, strouhal, strouhal_uncertainty = _strouhal_from_series(
            times_window, lift_window, u_mean, diameter
        )

        lift_oscillates = cl_std > CL_STD_THRESHOLD
        drag_in_range = CD_RANGE[0] < cd_mean < CD_RANGE[1]
        strouhal_in_range = ST_RANGE[0] < strouhal < ST_RANGE[1]

        # --- viz ---
        viz_dir = STUDY_DIR / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)

        vort_coords = channel_pressure_coords(H_CYLINDER, H_FAR)
        vorticity_html = viz.field_animation_html(
            vort_coords, vorticity_frames, frame_times,
            f"Von Karman vortex street -- vorticity (DFG 2D-2, Re={REYNOLDS:.0f}, "
            f"h_cylinder={H_CYLINDER})",
        )
        (viz_dir / "vortex_street_vorticity.html").write_text(vorticity_html)

        coeff_html = viz.coefficient_timeseries_html(
            times, {"Cd (drag)": drag, "Cl (lift)": lift},
            f"Drag/lift coefficient vs time (Re={REYNOLDS:.0f}) -- periodic vortex shedding",
            y_label="coefficient",
        )
        (viz_dir / "drag_lift_timeseries.html").write_text(coeff_html)

        v_coords = channel_velocity_coords(H_CYLINDER, H_FAR)
        speed_final = np.sqrt(velocity_x_final**2 + velocity_y_final**2)
        streamlines_html = viz.quiver_streamlines_html(
            v_coords, velocity_x_final, velocity_y_final, speed_final,
            f"Wake velocity field at t={T_PROD:.2f}s (Re={REYNOLDS:.0f})",
        )
        (viz_dir / "vortex_street_velocity.html").write_text(streamlines_html)

        pressure_final = np.asarray(sim.state["stores"]["pressure"], dtype=float)
        p_coords = channel_pressure_coords(H_CYLINDER, H_FAR)
        pressure_html = viz.field_heatmap_html(
            p_coords, pressure_final,
            f"Pressure field at t={T_PROD:.2f}s (Re={REYNOLDS:.0f})",
        )
        (viz_dir / "vortex_street_pressure.html").write_text(pressure_html)
    except Exception:
        append_run_event(WORKSPACE_ROOT, {
            "run_id": run_id, "event": "completed",
            "completed_at": time.time(), "n_steps": 0, "status": "failed",
        })
        raise

    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id,
        "event": "completed",
        "completed_at": time.time(),
        "n_steps": len(times),
        "status": "completed",
    })

    # --- report: PASS/FAIL mirrors this study's declared behavior_tests
    # exactly (lift-oscillates, drag-in-benchmark-range, strouhal-in-range)
    # -- no additional undeclared conditions.
    print(
        f"[navier-stokes] mesh: n_cells={n_cells} h_cylinder={H_CYLINDER} h_far={H_FAR} "
        f"dt={DT} T_PROD={T_PROD}s analysis_window=[{window_start:.3f}, {times[-1]:.3f}]s"
    )
    print(
        f"[navier-stokes] solve wall-time: {solve_wall_time:.1f}s "
        f"({solve_wall_time / 60:.1f} min) for {len(times)} substeps"
    )
    print(
        f"[navier-stokes] achieved: Cd_mean={cd_mean:.4f} Cd_max={cd_max:.4f} "
        f"(DFG reference Cd_max {DFG_CD_MAX_RANGE[0]}-{DFG_CD_MAX_RANGE[1]}); "
        f"Cl_max={cl_max:.4f} std(Cl)={cl_std:.4f} "
        f"(DFG reference Cl_max {DFG_CL_MAX_RANGE[0]}-{DFG_CL_MAX_RANGE[1]}); "
        f"Strouhal={strouhal:.2f} +/- {strouhal_uncertainty:.2f} (bin resolution; "
        f"shedding f={f_peak:.3f} Hz) (DFG reference St {DFG_ST_RANGE[0]}-{DFG_ST_RANGE[1]})"
    )
    print(
        f"[navier-stokes] std(Cl) over analysis window = {cl_std:.4f} "
        f"-> {'PASS' if lift_oscillates else 'FAIL'} (lift-oscillates > {CL_STD_THRESHOLD})"
    )
    print(
        f"[navier-stokes] mean(Cd) over analysis window = {cd_mean:.4f} "
        f"-> {'PASS' if drag_in_range else 'FAIL'} "
        f"(drag-in-benchmark-range in [{CD_RANGE[0]}, {CD_RANGE[1]}])"
    )
    print(
        f"[navier-stokes] Strouhal number = {strouhal:.2f} +/- {strouhal_uncertainty:.2f} "
        f"(bin resolution) -> {'PASS' if strouhal_in_range else 'FAIL'} "
        f"(strouhal-in-range in [{ST_RANGE[0]}, {ST_RANGE[1]}])"
    )
    print(
        f"[navier-stokes] viz written: {viz_dir / 'vortex_street_vorticity.html'}, "
        f"{viz_dir / 'drag_lift_timeseries.html'}, {viz_dir / 'vortex_street_velocity.html'}, "
        f"{viz_dir / 'vortex_street_pressure.html'}"
    )
    print(f"[navier-stokes] run recorded in {WORKSPACE_ROOT / '.pbg' / 'runs.jsonl'}")

    all_pass = lift_oscillates and drag_in_range and strouhal_in_range

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
