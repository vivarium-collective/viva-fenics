#!/usr/bin/env python3
"""Canonical run for the ``navier-stokes`` study (fluid-dynamics showcase).

Builds the ``navier_stokes`` composite (a single ``NavierStokesProcess`` --
real dolfinx IPCS lid-driven-cavity solve -- wired to a ``RAMEmitter``) and:

1. Runs the baseline (Re=100, resolution=24) to quasi-steady state, then a
   further short interval, to check the flow has stopped changing much
   (mirrors ``tests/test_flow.py::test_reaches_quasi_steady_state``).
2. Computes the mean absolute divergence of the converged velocity field
   (mirrors ``tests/test_flow.py::test_approximately_divergence_free``).
3. Runs the reynolds=[100,400,1000] sweep and checks every variant's final
   max speed stays finite and bounded (no blow-up) -- locking the stability
   finding from development (Re=1000 did NOT need to be dropped).
4. Renders the baseline's converged velocity field as an interactive
   streamline/quiver plot and the pressure field as a heatmap.

Standalone; run from the workspace root::

    pixi run python studies/navier-stokes/sims/run.py
"""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

import numpy as np
from process_bigraph import Composite, gather_emitter_results

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[1]

from viva_fenics import viz
from viva_fenics.core import build_core
from viva_fenics.composites.flow import navier_stokes
from viva_fenics.processes.flow import (
    _build_spaces,
    divergence_stats,
    pressure_coords,
    velocity_coords,
)
from vivarium_workbench.lib.run_log import append_run_event

SPEC_ID = "viva_fenics.composites.flow.navier_stokes"
STUDY_SLUG = "navier-stokes"
INVESTIGATION_SLUG = "fenics-showcase"

RESOLUTION = 24
DT = 0.01
LID_VELOCITY = 1.0  # NavierStokesProcess config default; navier_stokes() doesn't expose it as a sweep param
STEADY_TIME = 0.5
EXTRA_TIME = 0.1
REYNOLDS_BASELINE = 100.0
REYNOLDS_SWEEP = [100.0, 400.0, 1000.0]

QUASI_STEADY_TOL = 0.05
DIVERGENCE_TOL = 0.05
MAX_SPEED_SWEEP_TOL = 2.0 * LID_VELOCITY


def _run(reynolds: float, run_time: float, extra_time: float = 0.0):
    core = build_core()
    doc = navier_stokes(core, resolution=RESOLUTION, reynolds=reynolds, dt=DT)
    sim = Composite({"state": doc}, core=core)
    sim.run(run_time)
    if extra_time:
        sim.run(extra_time)
    rows = gather_emitter_results(sim)[("emitter",)]
    return rows


def _last_nonempty(rows, key):
    for row in reversed(rows):
        value = row.get(key)
        if value is not None and (not hasattr(value, "__len__") or len(value) > 0):
            return value
    raise ValueError(f"no non-empty '{key}' row recorded")


def main() -> int:
    run_id = uuid.uuid4().hex
    started = time.time()
    n_steps_estimate = int(round((STEADY_TIME + EXTRA_TIME) / DT))
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
            "resolution": RESOLUTION,
            "reynolds": REYNOLDS_BASELINE,
            "dt": DT,
            "steady_time": STEADY_TIME,
            "reynolds_sweep": REYNOLDS_SWEEP,
        },
    })

    try:
        # --- 1+2: baseline to quasi-steady, then a further short interval ---
        rows = _run(REYNOLDS_BASELINE, STEADY_TIME, EXTRA_TIME)
        speeds = [row["speed_integral"] for row in rows if row.get("speed_integral")]
        if len(speeds) < 2:
            print("[navier-stokes] ERROR: fewer than 2 speed_integral readings recorded", file=sys.stderr)
            append_run_event(WORKSPACE_ROOT, {
                "run_id": run_id,
                "event": "completed",
                "completed_at": time.time(),
                "n_steps": len(speeds),
                "status": "failed",
            })
            return 1
        speed_before, speed_after = speeds[-2], speeds[-1]
        quasi_steady_rel_change = abs(speed_after - speed_before) / speed_before
        reaches_quasi_steady = quasi_steady_rel_change < QUASI_STEADY_TOL

        velocity_x = np.asarray(_last_nonempty(rows, "velocity_x"))
        velocity_y = np.asarray(_last_nonempty(rows, "velocity_y"))
        pressure = np.asarray(_last_nonempty(rows, "pressure"))
        speed = np.sqrt(velocity_x**2 + velocity_y**2)

        # --- 2: approximate incompressibility of the converged baseline field ---
        domain, V, Q = _build_spaces(RESOLUTION)
        u_array = np.empty(2 * len(velocity_x))
        u_array[0::2] = velocity_x
        u_array[1::2] = velocity_y
        mean_abs_divergence, max_abs_divergence = divergence_stats(domain, V, u_array)
        approximately_divergence_free = mean_abs_divergence < DIVERGENCE_TOL

        # --- 3: stability across the reynolds sweep ---
        max_speed_across_sweep = 0.0
        sweep_results = []
        for reynolds in REYNOLDS_SWEEP:
            if reynolds == REYNOLDS_BASELINE:
                sweep_speed = float(speed.max())
            else:
                sweep_rows = _run(reynolds, STEADY_TIME)
                vx = np.asarray(_last_nonempty(sweep_rows, "velocity_x"))
                vy = np.asarray(_last_nonempty(sweep_rows, "velocity_y"))
                sweep_speed = float(np.sqrt(vx**2 + vy**2).max())
            finite = np.isfinite(sweep_speed)
            sweep_results.append((reynolds, sweep_speed, finite))
            if finite:
                max_speed_across_sweep = max(max_speed_across_sweep, sweep_speed)
            else:
                max_speed_across_sweep = float("inf")
        stable_across_reynolds_sweep = (
            np.isfinite(max_speed_across_sweep) and max_speed_across_sweep < MAX_SPEED_SWEEP_TOL
        )

        # --- 4: viz ---
        viz_dir = STUDY_DIR / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)

        v_coords = velocity_coords(RESOLUTION)
        streamlines_html = viz.quiver_streamlines_html(
            v_coords, velocity_x, velocity_y, speed,
            f"Lid-driven cavity velocity (Re={REYNOLDS_BASELINE:.0f}, resolution={RESOLUTION}) -- IPCS",
        )
        (viz_dir / "cavity_flow.html").write_text(streamlines_html)

        p_coords = pressure_coords(RESOLUTION)
        pressure_html = viz.field_heatmap_html(
            p_coords, pressure,
            f"Lid-driven cavity pressure (Re={REYNOLDS_BASELINE:.0f}, resolution={RESOLUTION})",
        )
        (viz_dir / "cavity_pressure.html").write_text(pressure_html)
    except Exception:
        append_run_event(WORKSPACE_ROOT, {
            "run_id": run_id,
            "event": "completed",
            "completed_at": time.time(),
            "n_steps": 0,
            "status": "failed",
        })
        raise

    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id,
        "event": "completed",
        "completed_at": time.time(),
        "n_steps": len(speeds),
        "status": "completed",
    })

    # --- report: PASS/FAIL mirrors this study's declared behavior_tests
    # exactly (reaches-quasi-steady, approximately-divergence-free,
    # stable-across-reynolds-sweep) -- no additional undeclared conditions.
    print(
        f"[navier-stokes] baseline Re={REYNOLDS_BASELINE:.0f} resolution={RESOLUTION} dt={DT}: "
        f"speed_integral {speed_before:.6f} -> {speed_after:.6f} "
        f"(rel change {quasi_steady_rel_change:.4f}) "
        f"-> {'PASS' if reaches_quasi_steady else 'FAIL'} (reaches-quasi-steady < {QUASI_STEADY_TOL})"
    )
    print(
        f"[navier-stokes] mean|div(u)|={mean_abs_divergence:.6f} max|div(u)|={max_abs_divergence:.4f} "
        f"-> {'PASS' if approximately_divergence_free else 'FAIL'} "
        f"(approximately-divergence-free < {DIVERGENCE_TOL})"
    )
    sweep_str = ", ".join(f"Re={re_:.0f}->max_speed={sp:.4f}" for re_, sp, _fin in sweep_results)
    print(
        f"[navier-stokes] reynolds sweep: {sweep_str} "
        f"-> {'PASS' if stable_across_reynolds_sweep else 'FAIL'} "
        f"(stable-across-reynolds-sweep < {MAX_SPEED_SWEEP_TOL})"
    )
    print(f"[navier-stokes] viz written: {viz_dir / 'cavity_flow.html'}, {viz_dir / 'cavity_pressure.html'}")
    print(f"[navier-stokes] run recorded in {WORKSPACE_ROOT / '.pbg' / 'runs.jsonl'}")

    all_pass = reaches_quasi_steady and approximately_divergence_free and stable_across_reynolds_sweep

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
