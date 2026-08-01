#!/usr/bin/env python3
"""Canonical run for the ``poisson-validation`` study.

Builds the ``poisson_baseline`` composite (a single ``PoissonSolverStep``
wired to a ``RAMEmitter``), solves the steady Poisson MMS problem once, and
renders the numeric solution as an interactive heatmap.

Standalone; run from the workspace root::

    pixi run python studies/poisson-validation/sims/run.py
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
from viva_fenics.composites.poisson import poisson_baseline
from vivarium_workbench.lib.run_log import append_run_event

SPEC_ID = "viva_fenics.composites.poisson.poisson_baseline"
STUDY_SLUG = "poisson-validation"
INVESTIGATION_SLUG = "fenics-showcase"

RESOLUTION = 32
DEGREE = 2
L2_TOLERANCE = 1e-10


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
        "n_steps": 1,
        "emitter": "ram",
        "origin": "canonical_run",
        "study_slug": STUDY_SLUG,
        "investigation_slug": INVESTIGATION_SLUG,
        "params": {"resolution": RESOLUTION, "degree": DEGREE},
    })

    try:
        core = build_core()
        doc = poisson_baseline(core, resolution=RESOLUTION, degree=DEGREE)
        sim = Composite({"state": doc}, core=core)
        sim.run(0.0)  # PoissonSolverStep is stateless -- fires once at t=0

        rows = gather_emitter_results(sim)[("emitter",)]
        row = next(r for r in rows if r.get("l2_error"))
        l2_error = float(row["l2_error"])
        solution = row["solution"]

        _, V = fem.build_mesh("unit_square", RESOLUTION, degree=DEGREE)
        coords = fem.node_coords(V)

        viz_dir = STUDY_DIR / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        html = viz.field_heatmap_html(
            coords, solution,
            f"Poisson MMS solution (resolution={RESOLUTION}, degree={DEGREE})",
        )
        (viz_dir / "solution_heatmap.html").write_text(html)
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
        "n_steps": 1,
        "status": "completed",
    })

    passed = l2_error < L2_TOLERANCE
    print(
        f"[poisson-validation] resolution={RESOLUTION} degree={DEGREE} "
        f"l2_error={l2_error:.3e} tol={L2_TOLERANCE:.0e} "
        f"-> {'PASS' if passed else 'FAIL'} (l2-error-within-tolerance)"
    )
    print(f"[poisson-validation] viz written: {viz_dir / 'solution_heatmap.html'}")
    print(f"[poisson-validation] run recorded in {WORKSPACE_ROOT / '.pbg' / 'runs.jsonl'}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
