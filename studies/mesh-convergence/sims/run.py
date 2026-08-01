#!/usr/bin/env python3
"""Canonical run for the ``mesh-convergence`` study.

Sweeps the ``mesh_convergence`` composite (one ``PoissonSolverStep`` per
point) over ``resolution in (8, 16, 32)`` at a fixed polynomial degree,
fits the observed L2-error vs mesh-size slope, and renders a log-log
convergence plot.

Standalone; run from the workspace root::

    pixi run python studies/mesh-convergence/sims/run.py
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np
from process_bigraph import Composite, gather_emitter_results

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[1]

from viva_fenics import viz
from viva_fenics.core import build_core
from viva_fenics.composites.convergence import mesh_convergence
from vivarium_workbench.lib.run_log import append_run_event

SPEC_ID = "viva_fenics.composites.convergence.mesh_convergence"
STUDY_SLUG = "mesh-convergence"
INVESTIGATION_SLUG = "fenics-showcase"

RESOLUTIONS = [8, 16, 32]
DEGREE = 1
MIN_RATE = 1.7  # P1 -> O(h^2); same band as tests/test_poisson.py::test_convergence_rate


def _l2_error_at(resolution: int) -> float:
    core = build_core()
    doc = mesh_convergence(core, resolution=resolution, degree=DEGREE)
    sim = Composite({"state": doc}, core=core)
    sim.run(0.0)  # PoissonSolverStep is stateless -- fires once at t=0
    rows = gather_emitter_results(sim)[("emitter",)]
    row = next(r for r in rows if r.get("l2_error"))
    return float(row["l2_error"])


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
        "n_steps": len(RESOLUTIONS),
        "emitter": "ram",
        "origin": "canonical_run",
        "study_slug": STUDY_SLUG,
        "investigation_slug": INVESTIGATION_SLUG,
        "params": {"resolutions": RESOLUTIONS, "degree": DEGREE},
    })

    try:
        errors = [_l2_error_at(res) for res in RESOLUTIONS]
        h = [1.0 / res for res in RESOLUTIONS]

        slope, _intercept = np.polyfit(np.log(h), np.log(errors), 1)
        rate = float(slope)

        viz_dir = STUDY_DIR / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        html = viz.convergence_loglog_html(h, errors)
        (viz_dir / "convergence.html").write_text(html)
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
        "n_steps": len(RESOLUTIONS),
        "status": "completed",
    })

    passed = rate >= MIN_RATE
    print(
        f"[mesh-convergence] resolutions={RESOLUTIONS} degree={DEGREE} "
        f"errors={[f'{e:.3e}' for e in errors]} rate={rate:.2f} "
        f"-> {'PASS' if passed else 'FAIL'} (convergence-rate-matches-order >= {MIN_RATE})"
    )
    print(f"[mesh-convergence] viz written: {viz_dir / 'convergence.html'}")
    print(f"[mesh-convergence] run recorded in {WORKSPACE_ROOT / '.pbg' / 'runs.jsonl'}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
