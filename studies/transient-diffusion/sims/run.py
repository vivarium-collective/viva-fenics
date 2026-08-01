#!/usr/bin/env python3
"""Canonical run for the ``transient-diffusion`` study.

Builds the ``transient_diffusion`` composite (a single ``DiffusionProcess``
advancing a gaussian bump by backward-Euler steps, wired to a
``RAMEmitter``), runs it for several ticks, and renders the field's decay
as an interactive time-slider animation.

Standalone; run from the workspace root::

    pixi run python studies/transient-diffusion/sims/run.py
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

from process_bigraph import Composite, gather_emitter_results

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[1]

from viva_fenics import fem, viz
from viva_fenics.core import build_core
from viva_fenics.composites.diffusion import transient_diffusion
from vivarium_workbench.lib.run_log import append_run_event

SPEC_ID = "viva_fenics.composites.diffusion.transient_diffusion"
STUDY_SLUG = "transient-diffusion"
INVESTIGATION_SLUG = "fenics-showcase"

RESOLUTION = 32
D = 0.1
DT = 0.01
# transient_diffusion's DiffusionProcess node has no explicit "interval" in
# the composite doc, so it uses process-bigraph's default tick interval
# (1.0) -- see tests/test_diffusion.py's
# test_diffusion_composite_integral_does_not_accumulate docstring.
TICK_INTERVAL = 1.0
N_TICKS = 4
RUN_TIME = N_TICKS * TICK_INTERVAL

MASS_DRIFT_TOLERANCE = 0.05


def main() -> int:
    run_id = uuid.uuid4().hex
    started = time.time()
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id,
        "event": "started",
        "spec_id": SPEC_ID,
        "label": STUDY_SLUG,
        "started_at": started,
        "status": "running",
        "n_steps": N_TICKS,
        "emitter": "ram",
        "origin": "canonical_run",
        "study_slug": STUDY_SLUG,
        "investigation_slug": INVESTIGATION_SLUG,
        "params": {"resolution": RESOLUTION, "D": D, "dt": DT, "run_time": RUN_TIME},
    })

    try:
        core = build_core()
        doc = transient_diffusion(core, resolution=RESOLUTION, D=D, dt=DT)
        sim = Composite({"state": doc}, core=core)
        sim.run(RUN_TIME)

        rows = gather_emitter_results(sim)[("emitter",)]
        solutions = [
            row["solution"] for row in rows
            if row.get("solution") is not None and len(row["solution"]) > 0
        ]
        # The emitter fires once before the process has ticked at all (so
        # "integral" is still its unset 0.0 default) and once per tick
        # thereafter; filter on truthiness to isolate post-tick readings (same
        # convention as tests/test_diffusion.py).
        integrals = [row["integral"] for row in rows if row.get("integral")]

        _, V = fem.build_mesh("unit_square", RESOLUTION, degree=1)
        coords = fem.node_coords(V)
        times = [i * TICK_INTERVAL for i in range(len(solutions))]

        viz_dir = STUDY_DIR / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        html = viz.field_animation_html(
            coords, solutions, times,
            f"Transient diffusion of a gaussian bump (D={D}, dt={DT})",
        )
        (viz_dir / "diffusion_animation.html").write_text(html)

        peak0, peakN = max(solutions[0]), max(solutions[-1])
        peak_decays = peakN < peak0

        mass0, massN = integrals[0], integrals[-1]
        mass_drift = abs(massN - mass0) / abs(mass0)
        mass_conserved = mass_drift < MASS_DRIFT_TOLERANCE
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
        "n_steps": N_TICKS,
        "status": "completed",
    })

    print(
        f"[transient-diffusion] resolution={RESOLUTION} D={D} dt={DT} "
        f"ticks={N_TICKS} peak: {peak0:.4f} -> {peakN:.4f} "
        f"-> {'PASS' if peak_decays else 'FAIL'} (peak-decays)"
    )
    print(
        f"[transient-diffusion] mass (FEM integral): {mass0:.4f} -> {massN:.4f} "
        f"drift={mass_drift:.2%} -> {'PASS' if mass_conserved else 'FAIL'} "
        f"(mass-conserved < {MASS_DRIFT_TOLERANCE:.0%})"
    )
    print(f"[transient-diffusion] viz written: {viz_dir / 'diffusion_animation.html'}")
    print(f"[transient-diffusion] run recorded in {WORKSPACE_ROOT / '.pbg' / 'runs.jsonl'}")

    return 0 if (peak_decays and mass_conserved) else 1


if __name__ == "__main__":
    raise SystemExit(main())
