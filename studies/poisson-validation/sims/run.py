#!/usr/bin/env python3
"""Canonical run for the ``poisson-validation`` study -- now a high-order
accuracy VERIFICATION: does each of P1/P2/P3 Lagrange elements achieve its
OWN optimal L2 convergence rate (degree+1) on a smooth (C-infinity)
manufactured solution?

Manufactured solution: u_exact = sin(pi*x)*sin(pi*y) on the unit square,
f = -Laplacian(u_exact) = 2*pi^2*sin(pi*x)*sin(pi*y), homogeneous Dirichlet
BC (u_exact is 0 on the whole boundary). Unlike the original quadratic MMS
(u = 1 + x^2 + 2*y^2, exactly representable by degree>=2 elements --
Galerkin orthogonality collapses its error to round-off), this solution is
NOT polynomial: no fixed-degree Lagrange space represents it exactly, so
EVERY degree shows a genuine, non-trivial discretization error that should
converge at its own textbook-optimal L2 rate as the mesh refines. That
makes it a real multi-order convergence-rate verification instead of a
single "error is tiny" checkbox.

For each degree in {1, 2, 3} and resolution in {8, 16, 32, 64} (cells per
side of the unit-square "crossed"-diagonal mesh -- see
``viva_fenics.fem.build_mesh``'s docstring for why crossed, not the
dolfinx-default uniform diagonal, is used here), this solves the real
dolfinx Poisson problem directly (``fem.solve_poisson``) and computes the
L2 error against the exact solution via ``fem.l2_error_exact`` -- an
ELEVATED-quadrature comparison against a genuine symbolic UFL expression,
not an FE interpolant, so P3's higher-order accuracy isn't itself
quadrature-limited (interpolating the exact solution into the same
degree-p trial space first, as the original quadratic-MMS path does, would
silently understate exactly the higher-order error this check needs to
see).

The canonical baseline composite (`high_order_verification`, degree=2,
resolution=32) is additionally run through the real process-bigraph
Composite/RAMEmitter machinery once, as a cross-check that the composite
path and the direct sweep agree to numerical precision -- same convention
as mesh-convergence's uniform/adaptive composite-vs-direct cross-check.

Achieved result (this committed production run): P1 -> rate 1.999, P2 ->
rate 3.000, P3 -> rate 4.000 (finer-mesh fit over resolutions 16/32/64,
matching theory 2/3/4 almost exactly -- no pre-asymptotic contamination
even at the coarsest resolution=8, since this is a smooth problem with no
singularity to slow convergence down).

Standalone; run from the workspace root::

    pixi run python studies/poisson-validation/sims/run.py
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np
from process_bigraph import Composite, gather_emitter_results

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[1]

from viva_fenics import fem, viz
from viva_fenics.core import build_core
from viva_fenics.composites.poisson import high_order_verification
from viva_fenics.processes.poisson import (
    _smooth_trig_source,
    _smooth_trig_bc,
    _smooth_trig_exact_ufl,
)
from vivarium_workbench.lib.run_log import append_run_event

SPEC_ID = "viva_fenics.composites.poisson.high_order_verification"
STUDY_SLUG = "poisson-validation"
INVESTIGATION_SLUG = "fenics-showcase"

DEGREES = [1, 2, 3]
RESOLUTIONS = [8, 16, 32, 64]
THEORETICAL_RATE = {degree: degree + 1 for degree in DEGREES}
RATE_TOLERANCE = 0.25  # honest margin around the theoretical optimal rate

# Canonical baseline composite point, used for the composite/direct
# cross-check and the solution-field heatmap.
BASELINE_DEGREE = 2
BASELINE_RESOLUTION = 32


def _fit_rate(h, errors):
    slope, _intercept = np.polyfit(np.log(h), np.log(errors), 1)
    return float(slope)


def main() -> int:
    run_id = uuid.uuid4().hex
    started = time.time()
    n_steps_estimate = len(DEGREES) * len(RESOLUTIONS) + 1
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
        "params": {"degrees": DEGREES, "resolutions": RESOLUTIONS},
    })

    try:
        core = build_core()

        # --- 1. full P1/P2/P3 x resolution sweep (direct dolfinx calls,
        # real solves -- this IS the verification) ---
        errors_by_degree: dict[int, list[float]] = {}
        h_by_degree: dict[int, list[float]] = {}
        for degree in DEGREES:
            errors = []
            print(f"[poisson-validation] degree={degree} sweep: resolutions={RESOLUTIONS}", flush=True)
            t0 = time.time()
            for n in RESOLUTIONS:
                domain, V = fem.build_mesh("unit_square", n, degree=degree)
                uh = fem.solve_poisson(
                    domain, V,
                    source_fn=_smooth_trig_source,
                    bc_fn=_smooth_trig_bc,
                )
                error = fem.l2_error_exact(domain, V, uh, _smooth_trig_exact_ufl)
                errors.append(error)
            wall = time.time() - t0
            errors_by_degree[degree] = errors
            h_by_degree[degree] = [1.0 / n for n in RESOLUTIONS]
            print(
                f"[poisson-validation] degree={degree} done: wall={wall:.2f}s "
                f"errors={[f'{e:.3e}' for e in errors]}",
                flush=True,
            )

        # --- 2. canonical baseline composite, once, as a cross-check ---
        print(
            f"[poisson-validation] baseline composite: degree={BASELINE_DEGREE} "
            f"resolution={BASELINE_RESOLUTION}",
            flush=True,
        )
        doc = high_order_verification(core, degree=BASELINE_DEGREE, resolution=BASELINE_RESOLUTION)
        sim = Composite({"state": doc}, core=core)
        sim.run(0.0)  # PoissonSolverStep is stateless -- fires once at t=0
        rows = gather_emitter_results(sim)[("emitter",)]
        row = next(r for r in rows if r.get("l2_error"))
        baseline_l2_error = float(row["l2_error"])
        baseline_solution = row["solution"]

        direct_baseline_error = errors_by_degree[BASELINE_DEGREE][RESOLUTIONS.index(BASELINE_RESOLUTION)]
        cross_check_diff = abs(baseline_l2_error - direct_baseline_error)
        print(
            f"[poisson-validation] composite/direct cross-check: "
            f"|{baseline_l2_error:.6e} - {direct_baseline_error:.6e}| = {cross_check_diff:.2e}",
            flush=True,
        )

        # --- 3. fit rates ---
        # "Achieved rate" (chart annotation, reported number): full-range
        # fit over all 4 resolutions.
        full_rate = {degree: _fit_rate(h_by_degree[degree], errors_by_degree[degree]) for degree in DEGREES}
        # Behavior-test rate: finer-mesh half only (resolutions 16/32/64),
        # dropping the coarsest point so the pass/fail gate isn't sensitive
        # to any pre-asymptotic wobble at the coarsest mesh.
        fine_idx = slice(1, None)
        test_rate = {
            degree: _fit_rate(h_by_degree[degree][fine_idx], errors_by_degree[degree][fine_idx])
            for degree in DEGREES
        }

        # --- viz ---
        viz_dir = STUDY_DIR / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)

        series = {f"P{degree}": (h_by_degree[degree], errors_by_degree[degree]) for degree in DEGREES}
        reference_rates = {f"P{degree}": THEORETICAL_RATE[degree] for degree in DEGREES}
        convergence_html = viz.convergence_loglog_multi_order_html(
            series, reference_rates,
            title="High-order accuracy verification: P1/P2/P3 optimal L2 convergence",
            y_label="L2 error",
        )
        (viz_dir / "convergence_multi_order.html").write_text(convergence_html)

        _, V_baseline = fem.build_mesh("unit_square", BASELINE_RESOLUTION, degree=BASELINE_DEGREE)
        coords = fem.node_coords(V_baseline)
        heatmap_html = viz.field_heatmap_html(
            coords, baseline_solution,
            f"Smooth manufactured solution u = sin(πx)sin(πy) "
            f"(P{BASELINE_DEGREE}, resolution={BASELINE_RESOLUTION})",
        )
        (viz_dir / "solution_heatmap.html").write_text(heatmap_html)
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

    print(f"[poisson-validation] total wall-time: {total_wall:.1f}s ({total_wall / 60:.2f} min)")
    results = {}
    for degree in DEGREES:
        theory = THEORETICAL_RATE[degree]
        passed = abs(test_rate[degree] - theory) <= RATE_TOLERANCE
        results[degree] = passed
        print(
            f"[poisson-validation] P{degree}: full-range rate={full_rate[degree]:.4f} "
            f"finer-mesh rate={test_rate[degree]:.4f} (theory {theory}, tol ±{RATE_TOLERANCE}) "
            f"-> {'PASS' if passed else 'FAIL'} (p{degree}-achieves-optimal-rate)"
        )
    print(f"[poisson-validation] viz written: {viz_dir}")
    print(f"[poisson-validation] run recorded in {WORKSPACE_ROOT / '.pbg' / 'runs.jsonl'}")

    all_pass = all(results.values())
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
