"""Behavior tests for gmsh-generated non-trivial-domain FEM (real gmsh +
real dolfinx import + real Poisson solve -- see viva_fenics/fem_gmsh.py and
viva_fenics/processes/complex_geometry.py's module docstrings).

Small resolution for pytest speed; see studies/complex-geometry/sims/run.py
for the full canonical run.
"""

import numpy as np
from process_bigraph import Composite, allocate_core, gather_emitter_results

from viva_fenics import fem_gmsh
from viva_fenics.processes.complex_geometry import ComplexGeometryStep

RESOLUTION = 12


# ---------------------------------------------------------------------------
# (a) each gmsh geometry imports to a dolfinx mesh with an expected positive
#     #cells
# ---------------------------------------------------------------------------

def test_all_geometries_import_with_positive_cell_count():
    for geometry in fem_gmsh.GEOMETRIES:
        domain, V = fem_gmsh.build_gmsh_mesh(geometry, RESOLUTION)
        n_cells = fem_gmsh.n_cells(domain)
        assert n_cells > 0, f"{geometry}: expected a non-trivial triangulation, got {n_cells} cells"
        # Sanity floor: a resolution=12 mesh over a domain with linear extent
        # ~1 should produce well more than a handful of triangles -- catches
        # a "degenerate/empty geometry silently imported" regression.
        assert n_cells > 20, f"{geometry}: suspiciously few cells ({n_cells}) for resolution={RESOLUTION}"
        assert V.dofmap.index_map.size_local > 0


def test_obstacle_mesh_excludes_the_hole_interior():
    """The obstacle geometry's mesh vertices must avoid the circular hole
    (radius 0.2, centered at (0.5, 0.5)) -- confirms the boolean cut
    actually removed that region, not just that *some* mesh was produced.
    """
    domain, V = fem_gmsh.build_gmsh_mesh("obstacle", RESOLUTION)
    coords = fem_gmsh.node_coords(V)
    dist_from_center = np.hypot(coords[:, 0] - 0.5, coords[:, 1] - 0.5)
    assert np.all(dist_from_center >= 0.2 - 1e-9), (
        "found a mesh node strictly inside the obstacle's circular hole "
        "-- the boolean cut did not remove it"
    )


def test_annulus_mesh_is_multiply_connected_hole():
    """The annulus geometry's mesh vertices must all lie between the inner
    (r=0.2) and outer (r=0.5) radii -- confirms both boundary components
    (inner hole + outer edge) are real, not just the outer disk.
    """
    domain, V = fem_gmsh.build_gmsh_mesh("annulus", RESOLUTION)
    coords = fem_gmsh.node_coords(V)
    r = np.hypot(coords[:, 0], coords[:, 1])
    assert np.all(r >= 0.2 - 1e-9), "found a node inside the annulus's inner hole"
    assert np.all(r <= 0.5 + 1e-9), "found a node outside the annulus's outer radius"


def test_lshape_mesh_excludes_the_notched_quadrant():
    """The L-shape geometry's mesh vertices must avoid the removed
    upper-right quadrant [0.5,1]x[0.5,1] (excluding its boundary edges)."""
    domain, V = fem_gmsh.build_gmsh_mesh("lshape", RESOLUTION)
    coords = fem_gmsh.node_coords(V)
    in_notch = (coords[:, 0] > 0.5 + 1e-9) & (coords[:, 1] > 0.5 + 1e-9)
    assert not np.any(in_notch), "found a mesh node strictly inside the notched-out quadrant"


# ---------------------------------------------------------------------------
# (b) a Poisson solve on each returns a finite, bounded, non-trivial
#     solution
# ---------------------------------------------------------------------------

def test_poisson_solve_finite_bounded_nontrivial_on_each_geometry():
    for geometry in fem_gmsh.GEOMETRIES:
        domain, V = fem_gmsh.build_gmsh_mesh(geometry, RESOLUTION)
        uh = fem_gmsh.solve_poisson_gmsh(domain, V, source_value=1.0)

        assert np.all(np.isfinite(uh)), f"{geometry}: solution has non-finite values"
        assert np.max(np.abs(uh)) < 1.0, f"{geometry}: solution unexpectedly large (unbounded?)"
        assert np.max(uh) > 1e-6, f"{geometry}: solution is trivially zero everywhere"
        # Homogeneous Dirichlet BC (u=0) on the full boundary + a positive
        # interior source -> the solution must be non-negative everywhere
        # (maximum principle for -Laplacian(u) = const > 0, u|boundary = 0).
        assert np.min(uh) > -1e-8, f"{geometry}: solution went negative, violating the maximum principle"


def test_solution_peaks_in_the_interior_away_from_boundary():
    """The obstacle solution's max should sit on an interior node, not on
    the boundary (where u is pinned to 0) -- confirms the Dirichlet BC was
    actually applied to the true exterior facets, not silently skipped."""
    domain, V = fem_gmsh.build_gmsh_mesh("obstacle", RESOLUTION)
    uh = fem_gmsh.solve_poisson_gmsh(domain, V, source_value=1.0)
    coords = fem_gmsh.node_coords(V)
    peak_idx = int(np.argmax(uh))
    peak_x, peak_y = coords[peak_idx]
    # Boundary consists of the unit-square edges (x/y in {0,1}) and the
    # circular hole (dist from (0.5,0.5) == 0.2); the peak should be well
    # clear of both.
    dist_from_hole = np.hypot(peak_x - 0.5, peak_y - 0.5)
    assert 0.01 < peak_x < 0.99 and 0.01 < peak_y < 0.99
    assert dist_from_hole > 0.25


# ---------------------------------------------------------------------------
# ComplexGeometryStep (the process-bigraph wrapper)
# ---------------------------------------------------------------------------

def test_complex_geometry_step_update():
    core = allocate_core()
    step = ComplexGeometryStep(
        config={"geometry": "obstacle", "resolution": RESOLUTION, "degree": 1},
        core=core,
    )
    out = step.update({})
    assert out["n_cells"] > 0
    assert len(out["solution"]) > 0
    assert np.isfinite(out["solution_max"])
    assert out["solution_max"] > 0
    assert np.all(np.isfinite(np.asarray(out["solution"])))


def test_complex_geometry_step_each_geometry_choice():
    core = allocate_core()
    for geometry in fem_gmsh.GEOMETRIES:
        step = ComplexGeometryStep(
            config={"geometry": geometry, "resolution": RESOLUTION, "degree": 1},
            core=core,
        )
        out = step.update({})
        assert out["n_cells"] > 0, f"{geometry}: n_cells should be positive"
        assert np.isfinite(out["solution_max"])


# ---------------------------------------------------------------------------
# (c) generator registered + full composite build/run
# ---------------------------------------------------------------------------

def test_complex_geometry_generator_registered():
    from viva_superpowers.composite_generator import _REGISTRY
    assert any(e.endswith(".complex_geometry") for e in _REGISTRY)


def test_complex_geometry_composite_runs_and_emits():
    from viva_fenics.core import build_core
    from viva_fenics.composites.complex_geometry import complex_geometry

    core = build_core()
    document = complex_geometry(core=core, geometry="lshape", resolution=RESOLUTION)
    sim = Composite({"state": document}, core=core)
    sim.run(0)  # Step-only composite: a single triggered update is enough

    rows = gather_emitter_results(sim)[("emitter",)]
    assert len(rows) >= 1
    last = rows[-1]
    assert last["n_cells"] > 0
    assert len(last["solution"]) > 0
    assert np.isfinite(last["solution_max"])
