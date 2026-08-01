"""Fast tests for the AMR (adaptive mesh refinement) L-shaped-corner
singularity module (``viva_fenics.fem_amr``) and its Step/composite wrapper.

Kept deliberately small (coarse initial mesh, few refinement levels) so the
whole file runs in well under a second of real dolfinx solves -- see
``studies/mesh-convergence/sims/run.py`` for the full production-sized
uniform-vs-adaptive comparison these thresholds are calibrated with headroom
below (see that study's report for the achieved production slopes:
uniform ~-0.338 vs theory -1/3, adaptive ~-0.528 vs theory -1/2).
"""

import numpy as np
from dolfinx import mesh as dmesh
from process_bigraph import Composite, allocate_core, gather_emitter_results

from viva_fenics import fem_amr


# Small/fast config shared by several tests below -- NOT the production
# sizing (see run.py), just enough levels for the qualitative signal
# (error decreasing, indicators concentrating, adaptive beating uniform) to
# already be unambiguous, per this module's development spike.
FAST_H = 0.35
FAST_LEVELS = 4


def _density_ratio(domain, r_near=0.15, r_far=0.5):
    """Cell-centroid density (cells / area) within r_near of the re-entrant
    corner, divided by the density beyond r_far -- >1 means refinement has
    concentrated more elements near the corner than far away, normalized
    for the different areas of the two regions (see module docstring's
    R_dT / density note in the caller for the exact areas used)."""
    tdim = domain.topology.dim
    ncells = domain.topology.index_map(tdim).size_local
    mids = dmesh.compute_midpoints(domain, tdim, np.arange(ncells, dtype=np.int32))
    dist = np.hypot(mids[:, 0], mids[:, 1])
    near = int((dist < r_near).sum())
    far = int((dist > r_far).sum())
    # The AMR domain is a 2x2 square minus a 1x1 quadrant -> area 3; a disk
    # of radius r centered at the re-entrant corner intersects the domain
    # in exactly 3/4 of the full disk (the domain's 270-degree wedge there).
    domain_area = 3.0
    area_near = np.pi * r_near**2 * 0.75
    area_far = domain_area - np.pi * r_far**2 * 0.75
    near_density = near / area_near
    far_density = far / area_far if far else 1e-12
    return near_density / far_density if far_density else float("inf")


def test_error_decreases_under_refinement():
    result = fem_amr.run_uniform_loop(FAST_H, n_levels=FAST_LEVELS, degree=1)
    errors = result["error_history"]
    assert len(errors) == FAST_LEVELS + 1
    # strictly decreasing at every level, not just net decrease
    assert all(errors[i + 1] < errors[i] for i in range(len(errors) - 1))
    assert errors[0] > 0.0


def test_estimator_concentrates_near_corner():
    # Finer than FAST_H specifically so at least a few cell centroids fall
    # within the r<0.15 "near corner" band being checked below (FAST_H=0.35
    # is coarse enough that zero cells land there) -- still a single cheap
    # solve (~190 cells).
    domain = fem_amr.build_lshape_amr_mesh(0.15)
    uh, V, _energy_error = fem_amr.solve_laplace_lshape(domain, degree=1)
    indicators = fem_amr.error_indicators(domain, uh)
    assert indicators.shape[0] == fem_amr.n_cells(domain)
    assert np.all(indicators >= 0.0)

    tdim = domain.topology.dim
    mids = dmesh.compute_midpoints(domain, tdim, np.arange(fem_amr.n_cells(domain), dtype=np.int32))
    dist = np.hypot(mids[:, 0], mids[:, 1])
    near = indicators[dist < 0.15]
    far = indicators[dist > 0.5]
    assert near.size > 0 and far.size > 0
    # residual (edge-jump) indicators near the singular corner must be
    # dramatically larger than in the smooth far-field -- the estimator's
    # whole job is distinguishing these; development spike measured ~150x.
    assert near.mean() > 20 * far.mean()


def test_dorfler_marking_is_bulk_chasing():
    indicators = np.array([100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    marked = fem_amr.dorfler_mark(indicators, theta=0.5)
    # a single dominant cell holding >50% of total squared error should be
    # marked alone -- Doerfler marking should NOT over-mark.
    assert list(marked) == [0]

    marked_all = fem_amr.dorfler_mark(indicators, theta=1.0)
    assert len(marked_all) == indicators.size

    empty = fem_amr.dorfler_mark(np.zeros(5), theta=0.5)
    assert empty.size == 0


def test_refine_marked_increases_cell_count():
    domain = fem_amr.build_lshape_amr_mesh(FAST_H)
    n0 = fem_amr.n_cells(domain)
    marked = np.array([0, 1], dtype=np.int32)
    refined = fem_amr.refine_marked(domain, marked)
    assert fem_amr.n_cells(refined) > n0


def test_refine_uniform_quadruples_roughly():
    domain = fem_amr.build_lshape_amr_mesh(FAST_H)
    n0 = fem_amr.n_cells(domain)
    refined = fem_amr.refine_uniform(domain)
    # 2D uniform (every-edge) bisection: exactly 4x for a conforming
    # triangulation with no hanging nodes to begin with.
    assert fem_amr.n_cells(refined) == 4 * n0


def test_adaptive_beats_uniform_slope():
    uniform = fem_amr.run_uniform_loop(FAST_H, n_levels=FAST_LEVELS, degree=1)
    adaptive = fem_amr.run_amr_loop(FAST_H, n_refinements=FAST_LEVELS, theta=0.5, degree=1)

    slope_u, _ = np.polyfit(np.log(uniform["dofs_history"]), np.log(uniform["error_history"]), 1)
    slope_a, _ = np.polyfit(np.log(adaptive["dofs_history"]), np.log(adaptive["error_history"]), 1)

    # both must be genuine convergence (negative slope: error falls as
    # dofs grow) ...
    assert slope_u < 0
    assert slope_a < 0
    # ... and adaptive must be CLEARLY steeper (more negative) than uniform
    # -- the qualitative signature of AMR recovering a better rate on a
    # singular problem. (Even at this short/coarse fast-test sizing the
    # separation is large -- development spike measured ~-0.68 vs ~-0.34;
    # 0.1 margin below that is generous.)
    assert slope_a < slope_u - 0.1


def test_refinement_concentrates_at_corner():
    initial = fem_amr.build_lshape_amr_mesh(FAST_H)
    ratio_initial = _density_ratio(initial)

    adaptive = fem_amr.run_amr_loop(FAST_H, n_refinements=FAST_LEVELS, theta=0.5, degree=1)
    ratio_final = _density_ratio(adaptive["domain"])

    # the adaptive mesh's near-corner cell density, relative to its
    # far-field density, must have grown substantially past the initial
    # (near-uniform) mesh's ratio -- direct evidence refinement concentrated
    # at the singularity rather than spreading uniformly.
    assert ratio_final > ratio_initial + 5.0


def test_run_amr_loop_captures_mesh_snapshots():
    result = fem_amr.run_amr_loop(FAST_H, n_refinements=3, theta=0.5, degree=1, capture_levels=[0, 2])
    assert set(result["snapshots"]) == {0, 2}
    for coords, cells in result["snapshots"].values():
        assert coords.ndim == 2 and coords.shape[1] == 2
        assert cells.ndim == 2 and cells.shape[1] == 3


def test_adaptive_refinement_step_runs():
    from viva_fenics.processes.amr import AdaptiveRefinementStep

    core = allocate_core()
    out = AdaptiveRefinementStep(
        config={"initial_h": FAST_H, "n_refinements": FAST_LEVELS, "marking_fraction": 0.5, "degree": 1},
        core=core,
    ).update({})
    assert out["energy_error"] > 0.0
    assert len(out["dofs_history"]) == FAST_LEVELS + 1
    assert len(out["error_history"]) == FAST_LEVELS + 1
    assert out["error_history"][-1] < out["error_history"][0]
    assert len(out["solution"]) == out["dofs_history"][-1]


def test_adaptive_refinement_generator_registered():
    from viva_superpowers.composite_generator import _REGISTRY

    assert any(e.endswith(".adaptive_refinement") for e in _REGISTRY)


def test_adaptive_refinement_builds():
    from viva_fenics.core import build_core
    from viva_fenics.composites.amr import adaptive_refinement

    core = build_core()
    doc = adaptive_refinement(core, n_refinements=FAST_LEVELS, marking_fraction=0.5, degree=1)
    sim = Composite({"state": doc}, core=core)
    sim.run(0.0)

    rows = gather_emitter_results(sim)[("emitter",)]
    row = next(r for r in rows if r.get("energy_error"))
    assert row["energy_error"] > 0.0
    assert len(row["error_history"]) == FAST_LEVELS + 1
