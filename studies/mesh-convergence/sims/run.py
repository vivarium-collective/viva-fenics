#!/usr/bin/env python3
"""Canonical run for the ``mesh-convergence`` study -- now the investigation's
Adaptive Mesh Refinement (AMR) showpiece: recovering near-optimal energy-norm
convergence on the classic L-shaped-domain re-entrant-corner Laplace
singularity, where UNIFORM refinement is provably capped at a suboptimal
rate.

Runs THREE real dolfinx computations (see ``viva_fenics.fem_amr`` for the
mesh/solve/estimator/marking/refine machinery -- all real, no faking):

1. A UNIFORM-refinement sequence (``fem_amr.run_uniform_loop``) -- direct,
   not composite-wrapped (no Step exists for it; the composite generator
   this investigation ships is specifically ``adaptive_refinement``).
2. The ADAPTIVE (AMR) sequence, run through the real
   ``adaptive_refinement`` composite (``AdaptiveRefinementStep``) -- this
   study's canonical baseline composite id, recorded via a RAMEmitter same
   as every other study in this investigation.
3. The SAME adaptive sequence again, called directly via
   ``fem_amr.run_amr_loop`` with ``capture_levels`` set -- needed because
   the Step's outputs are scalars only (dofs/error history + final
   solution), not per-level mesh geometry; this direct call additionally
   snapshots node coordinates + triangle connectivity at levels 0-8 for the
   refinement animation. Its dofs/error history is also used as the
   authoritative source for the convergence chart + density-concentration
   metric (deterministic dolfinx, LU direct solve, no MPI/threading
   nondeterminism -- so it reproduces (2)'s numbers exactly; (2) exists to
   genuinely exercise the composite/Step machinery, not because its
   numbers differ).

Standalone; run from the workspace root::

    pixi run python studies/mesh-convergence/sims/run.py

Runtime note: all three computations combined run in well under a minute on
this machine (uniform ~2s, adaptive-via-composite ~7s, adaptive-with-
snapshots ~7s) -- an AMR loop is a SEQUENCE of direct LU solves (a few
dozen total, largest ~260k dofs), not a long timestepped simulation, so
there is no wall-time pressure here even at fairly deep refinement levels.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np
from dolfinx import mesh as dmesh
from process_bigraph import Composite, gather_emitter_results

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[1]

from viva_fenics import fem_amr, viz
from viva_fenics.core import build_core
from viva_fenics.composites.amr import adaptive_refinement
from vivarium_workbench.lib.run_log import append_run_event

SPEC_ID = "viva_fenics.composites.amr.adaptive_refinement"
STUDY_SLUG = "mesh-convergence"
INVESTIGATION_SLUG = "fenics-showcase"

INITIAL_H = 0.3
UNIFORM_LEVELS = 6          # 7 solves: 69 -> 218,049 dofs
ADAPTIVE_REFINEMENTS = 15   # 16 solves: 69 -> 259,055 dofs
MARKING_FRACTION = 0.5      # Doerfler theta
DEGREE = 1
ANIMATION_CAPTURE_LEVELS = list(range(9))  # levels 0-8: 106 -> 7,879 cells (viz-size budget)

# Re-entrant corner "near"/"far" radii and the domain's true area (a 2x2
# square minus a 1x1 quadrant: area = 4 - 1 = 3) -- used to normalize raw
# cell counts into a density (a disk of radius r centered at the corner
# intersects this domain in exactly 3/4 of the full disk, the domain's
# 270-degree wedge there).
R_NEAR = 0.15
R_FAR = 0.5
DOMAIN_AREA = 3.0

# Behavior-test thresholds -- see study.yaml's expected_behavior/
# behavior_tests for the same numbers with rationale. Calibrated with
# headroom below the achieved production values printed below.
ADAPTIVE_RATE_MAX = -0.45          # near theory -1/2; achieved ~-0.528
RATE_SEPARATION_MIN = 0.10          # adaptive must be this much steeper than uniform
DENSITY_RATIO_MIN = 15.0            # achieved ~29.2
CONCENTRATION_GROWTH_MIN = 10.0     # achieved ~55.8x


def _density_ratio(domain, r_near=R_NEAR, r_far=R_FAR):
    """Near-corner / far-field cell-centroid density ratio -- see module
    docstring's radius/area note."""
    tdim = domain.topology.dim
    ncells = domain.topology.index_map(tdim).size_local
    mids = dmesh.compute_midpoints(domain, tdim, np.arange(ncells, dtype=np.int32))
    dist = np.hypot(mids[:, 0], mids[:, 1])
    near = int((dist < r_near).sum())
    far = int((dist > r_far).sum())
    area_near = np.pi * r_near**2 * 0.75
    area_far = DOMAIN_AREA - np.pi * r_far**2 * 0.75
    near_density = near / area_near
    far_density = far / area_far if far else 1e-12
    return (near_density / far_density) if far_density else float("inf")


def _fit_rate(dofs, errors):
    slope, _intercept = np.polyfit(np.log(dofs), np.log(errors), 1)
    return float(slope)


def main() -> int:
    run_id = uuid.uuid4().hex
    started = time.time()
    n_steps_estimate = (UNIFORM_LEVELS + 1) + 2 * (ADAPTIVE_REFINEMENTS + 1)
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
            "initial_h": INITIAL_H,
            "uniform_levels": UNIFORM_LEVELS,
            "adaptive_refinements": ADAPTIVE_REFINEMENTS,
            "marking_fraction": MARKING_FRACTION,
            "degree": DEGREE,
        },
    })

    try:
        core = build_core()

        # --- 1. uniform-refinement sequence (direct, no Step) ---
        print(f"[mesh-convergence] uniform refinement: initial_h={INITIAL_H} "
              f"n_levels={UNIFORM_LEVELS} degree={DEGREE}", flush=True)
        t0 = time.time()
        uniform = fem_amr.run_uniform_loop(INITIAL_H, n_levels=UNIFORM_LEVELS, degree=DEGREE)
        uniform_wall = time.time() - t0
        uniform_dofs = uniform["dofs_history"]
        uniform_errors = uniform["error_history"]
        print(
            f"[mesh-convergence] uniform done: wall={uniform_wall:.1f}s "
            f"dofs={[int(d) for d in uniform_dofs]}",
            flush=True,
        )

        # --- 2. adaptive sequence via the REAL composite (canonical run) ---
        print(f"[mesh-convergence] adaptive AMR (composite): n_refinements={ADAPTIVE_REFINEMENTS} "
              f"theta={MARKING_FRACTION} degree={DEGREE}", flush=True)
        t0 = time.time()
        doc = adaptive_refinement(
            core, n_refinements=ADAPTIVE_REFINEMENTS,
            marking_fraction=MARKING_FRACTION, degree=DEGREE,
        )
        sim = Composite({"state": doc}, core=core)
        sim.run(0.0)
        rows = gather_emitter_results(sim)[("emitter",)]
        composite_row = next(r for r in rows if r.get("energy_error"))
        composite_wall = time.time() - t0
        print(
            f"[mesh-convergence] adaptive composite done: wall={composite_wall:.1f}s "
            f"final_energy_error={composite_row['energy_error']:.4e}",
            flush=True,
        )

        # --- 3. same adaptive sequence, direct call, WITH mesh snapshots
        # (needed for the refinement animation + corner-density metric --
        # see module docstring for why this duplicates (2)'s solves) ---
        print(f"[mesh-convergence] adaptive AMR (direct, capturing mesh snapshots at "
              f"levels {ANIMATION_CAPTURE_LEVELS})...", flush=True)
        t0 = time.time()
        adaptive = fem_amr.run_amr_loop(
            INITIAL_H, n_refinements=ADAPTIVE_REFINEMENTS, theta=MARKING_FRACTION,
            degree=DEGREE, capture_levels=ANIMATION_CAPTURE_LEVELS,
        )
        adaptive_wall = time.time() - t0
        adaptive_dofs = adaptive["dofs_history"]
        adaptive_errors = adaptive["error_history"]
        print(
            f"[mesh-convergence] adaptive (direct) done: wall={adaptive_wall:.1f}s "
            f"dofs={[int(d) for d in adaptive_dofs]}",
            flush=True,
        )
        # sanity cross-check: the composite and the direct call ran the
        # identical deterministic computation, so their final energy errors
        # should agree to numerical precision.
        cross_check_diff = abs(composite_row["energy_error"] - adaptive_errors[-1])
        print(
            f"[mesh-convergence] composite/direct cross-check: "
            f"|{composite_row['energy_error']:.6e} - {adaptive_errors[-1]:.6e}| = {cross_check_diff:.2e}",
            flush=True,
        )

        # --- fit rates ---
        uniform_rate = _fit_rate(uniform_dofs, uniform_errors)
        adaptive_rate = _fit_rate(adaptive_dofs, adaptive_errors)

        # --- corner-concentration evidence ---
        initial_domain = fem_amr.build_lshape_amr_mesh(INITIAL_H)
        ratio_initial = _density_ratio(initial_domain)
        ratio_final = _density_ratio(adaptive["domain"])
        concentration_growth = (ratio_final / ratio_initial) if ratio_initial > 0 else float("inf")

        # --- viz ---
        viz_dir = STUDY_DIR / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)

        snapshots = adaptive["snapshots"]
        levels_sorted = sorted(snapshots)
        frames_coords = [snapshots[lvl][0] for lvl in levels_sorted]
        frames_cells = [snapshots[lvl][1] for lvl in levels_sorted]
        labels = [f"level {lvl}: {frames_cells[i].shape[0]} cells" for i, lvl in enumerate(levels_sorted)]
        mesh_anim_html = viz.mesh_wireframe_animation_html(
            frames_coords, frames_cells, labels,
            "Adaptive mesh refinement concentrating at the re-entrant corner "
            f"(Doerfler theta={MARKING_FRACTION}, degree={DEGREE})",
            highlight=fem_amr.REENTRANT_CORNER,
        )
        (viz_dir / "mesh_refinement_animation.html").write_text(mesh_anim_html)

        convergence_html = viz.convergence_loglog_compare_html(
            {"uniform": (uniform_dofs, uniform_errors), "adaptive (AMR)": (adaptive_dofs, adaptive_errors)},
            title="Energy-norm convergence: uniform vs adaptive (L-shaped corner singularity)",
        )
        (viz_dir / "convergence_comparison.html").write_text(convergence_html)

        final_coords = fem_amr.node_coords(adaptive["V"])
        final_solution_html = viz.field_heatmap_html(
            final_coords, adaptive["uh"].x.array,
            f"Final adaptive solution u = r^(2/3)sin(2θ/3) "
            f"(level {ADAPTIVE_REFINEMENTS}, {adaptive_dofs[-1]:.0f} dofs)",
        )
        (viz_dir / "final_solution_heatmap.html").write_text(final_solution_html)
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
        "n_steps": n_steps_estimate,
        "status": "completed",
    })

    total_wall = time.time() - started

    rate_beats = adaptive_rate <= uniform_rate - RATE_SEPARATION_MIN
    rate_near_optimal = adaptive_rate <= ADAPTIVE_RATE_MAX
    adaptive_beats_uniform = rate_beats and rate_near_optimal

    density_high = ratio_final >= DENSITY_RATIO_MIN
    grew_enough = concentration_growth >= CONCENTRATION_GROWTH_MIN
    concentrates_at_corner = density_high and grew_enough

    print(f"[mesh-convergence] total wall-time: {total_wall:.1f}s ({total_wall / 60:.1f} min)")
    print(
        f"[mesh-convergence] uniform rate={uniform_rate:.4f} (theory -1/3={-1/3:.4f}) "
        f"final_dofs={uniform_dofs[-1]:.0f} final_error={uniform_errors[-1]:.4e}"
    )
    print(
        f"[mesh-convergence] adaptive rate={adaptive_rate:.4f} (theory -1/2={-1/2:.4f}) "
        f"final_dofs={adaptive_dofs[-1]:.0f} final_error={adaptive_errors[-1]:.4e}"
    )
    print(
        f"[mesh-convergence] adaptive-beats-uniform-rate: separation={uniform_rate - adaptive_rate:.4f} "
        f"(need >= {RATE_SEPARATION_MIN}), adaptive_rate={adaptive_rate:.4f} (need <= {ADAPTIVE_RATE_MAX}) "
        f"-> {'PASS' if adaptive_beats_uniform else 'FAIL'}"
    )
    print(
        f"[mesh-convergence] corner density ratio: initial={ratio_initial:.3f} final={ratio_final:.3f} "
        f"growth={concentration_growth:.2f}x"
    )
    print(
        f"[mesh-convergence] refinement-concentrates-at-corner: final_ratio={ratio_final:.3f} "
        f"(need >= {DENSITY_RATIO_MIN}), growth={concentration_growth:.2f}x (need >= {CONCENTRATION_GROWTH_MIN}) "
        f"-> {'PASS' if concentrates_at_corner else 'FAIL'}"
    )
    print(f"[mesh-convergence] viz written: {viz_dir}")
    print(f"[mesh-convergence] run recorded in {WORKSPACE_ROOT / '.pbg' / 'runs.jsonl'}")

    all_pass = adaptive_beats_uniform and concentrates_at_corner
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
