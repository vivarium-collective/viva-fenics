#!/usr/bin/env python3
"""Canonical run for the ``reaction-diffusion`` study (the coupling
showcase).

Builds the ``reaction_diffusion`` composite -- ``DiffusionProcess`` and
``LogisticReactionProcess`` wired through shared ``stores.solution`` /
``stores.source`` bigraph stores, producing Fisher-KPP dynamics BY
COMPOSITION (neither process implements Fisher-KPP itself). Runs the
coupled composite (r=2.0) alongside a near-zero-reaction control (r=1e-9,
identical wiring) to confirm mass growth is driven by the coupling, not
solver noise -- mirrors ``tests/test_reaction_diffusion.py::
test_fisher_kpp_front_grows``. Renders the coupled run's wavefront as an
interactive time-slider animation.

Standalone; run from the workspace root::

    pixi run python studies/reaction-diffusion/sims/run.py
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
from viva_fenics.composites.reaction_diffusion import reaction_diffusion
from vivarium_workbench.lib.run_log import append_run_event

SPEC_ID = "viva_fenics.composites.reaction_diffusion.reaction_diffusion"
STUDY_SLUG = "reaction-diffusion"
INVESTIGATION_SLUG = "fenics-showcase"

RESOLUTION = 24
D = 0.05
R = 2.0
R_CONTROL = 1e-9  # near-zero reaction; genuine disconnect control (see test docstring)
DT = 0.01
RUN_TIME = 0.5
K = 1.0  # LogisticReactionProcess default carrying capacity

GROWTH_RATIO_MIN = 5.0
FIELD_BOUND = 1.2  # K=1.0 + margin

# Both process nodes in this composite set "interval": dt explicitly, so
# composite ticks happen every dt (not the process-bigraph default 1.0) --
# unlike transient_diffusion. Subsample the emitter so a 0.5s run at
# dt=0.01 (50 ticks) doesn't render an unwieldy 51-frame animation.
EMITTER_SUBSAMPLE = 5


def _run(r: float, run_time: float):
    core = build_core()
    doc = reaction_diffusion(core, resolution=RESOLUTION, D=D, r=r, dt=DT)
    doc["emitter"]["config"]["subsample"] = EMITTER_SUBSAMPLE
    sim = Composite({"state": doc}, core=core)
    sim.run(run_time)
    rows = gather_emitter_results(sim)[("emitter",)]
    integrals = [row["integral"] for row in rows if row.get("integral")]
    solutions = [
        row["solution"] for row in rows
        if row.get("solution") is not None and len(row["solution"]) > 0
    ]
    return integrals, solutions


def main() -> int:
    run_id = uuid.uuid4().hex
    started = time.time()
    n_steps_estimate = int(round(RUN_TIME / DT))
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
        "params": {"resolution": RESOLUTION, "D": D, "r": R, "dt": DT, "run_time": RUN_TIME},
    })

    try:
        integrals_main, solutions_main = _run(R, RUN_TIME)
        integrals_off, _solutions_off = _run(R_CONTROL, RUN_TIME)

        growth_main = integrals_main[-1] - integrals_main[0]
        growth_off = integrals_off[-1] - integrals_off[0]
        growth_ratio = growth_main / max(abs(growth_off), 1e-12)
        # PASS/FAIL mirrors this study's declared behavior_tests exactly (only
        # growth_ratio >= GROWTH_RATIO_MIN is asserted there) -- no additional
        # undeclared conditions.
        mass_grows = growth_ratio >= GROWTH_RATIO_MIN

        max_field = max(max(s) for s in solutions_main)
        field_bounded = max_field < FIELD_BOUND

        _, V = fem.build_mesh("unit_square", RESOLUTION, degree=1)
        coords = fem.node_coords(V)
        times = [i * DT * EMITTER_SUBSAMPLE for i in range(len(solutions_main))]

        viz_dir = STUDY_DIR / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        html = viz.field_animation_html(
            coords, solutions_main, times,
            f"Fisher-KPP wavefront (D={D}, r={R}, K={K}) -- Diffusion ⊕ LogisticReaction",
        )
        (viz_dir / "reaction_diffusion_animation.html").write_text(html)
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
        "n_steps": len(solutions_main),
        "status": "completed",
    })

    print(
        f"[reaction-diffusion] resolution={RESOLUTION} D={D} r={R} vs r_control={R_CONTROL} "
        f"mass growth: {growth_main:.4f} vs {growth_off:.6f} ratio={growth_ratio:.1f}x "
        f"-> {'PASS' if mass_grows else 'FAIL'} (mass-grows-via-coupling >= {GROWTH_RATIO_MIN}x)"
    )
    print(
        f"[reaction-diffusion] max field over run: {max_field:.4f} (K={K}) "
        f"-> {'PASS' if field_bounded else 'FAIL'} (field-bounded-by-k < {FIELD_BOUND})"
    )
    print(f"[reaction-diffusion] viz written: {viz_dir / 'reaction_diffusion_animation.html'}")
    print(f"[reaction-diffusion] run recorded in {WORKSPACE_ROOT / '.pbg' / 'runs.jsonl'}")

    return 0 if (mass_grows and field_bounded) else 1


if __name__ == "__main__":
    raise SystemExit(main())
