#!/usr/bin/env python3
"""Canonical run for the ``complex-geometry`` study (the gmsh
non-trivial-domain showcase).

Builds the ``complex_geometry`` composite (a single ``ComplexGeometryStep``
-- real gmsh geometry construction -> dolfinx import via
``dolfinx.io.gmsh.model_to_mesh`` -> Poisson solve, see
``viva_fenics/fem_gmsh.py``'s module docstring) for each of the three named
geometries (obstacle, lshape, annulus):

1. For every geometry, checks the imported mesh's cell count exceeds the
   declared floor (mirrors
   ``tests/test_complex_geometry.py::test_all_geometries_import_with_positive_cell_count``).
2. For every geometry, checks the Poisson solution is finite, non-negative,
   and reaches a max value strictly between 1e-6 and 1.0 (mirrors
   ``tests/test_complex_geometry.py::test_poisson_solve_finite_bounded_nontrivial_on_each_geometry``).
3. Renders the obstacle solution as a 2D field heatmap
   (``viz/field_obstacle.html``) and as a 3D Three.js mesh view, lifting the
   2D field into 3D via z=value (a "relief surface" on the true triangle
   connectivity of the imported mesh) colored by the same value
   (``viz/mesh3d.html``).

Standalone; run from the workspace root::

    pixi run python studies/complex-geometry/sims/run.py
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np
from process_bigraph import Composite, gather_emitter_results

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[1]

from viva_fenics import fem_gmsh, viz
from viva_fenics.core import build_core
from viva_fenics.composites.complex_geometry import complex_geometry
from vivarium_workbench.lib.run_log import append_run_event

SPEC_ID = "viva_fenics.composites.complex_geometry.complex_geometry"
STUDY_SLUG = "complex-geometry"
INVESTIGATION_SLUG = "fenics-showcase"

RESOLUTION = 32
GEOMETRIES = fem_gmsh.GEOMETRIES  # ("obstacle", "lshape", "annulus")

MIN_N_CELLS = 20
SOLUTION_MAX_FLOOR = 1e-6
SOLUTION_MAX_CEILING = 1.0


def _run_one(core, geometry, resolution=RESOLUTION):
    doc = complex_geometry(core, geometry=geometry, resolution=resolution)
    sim = Composite({"state": doc}, core=core)
    sim.run(0)  # ComplexGeometryStep is stateless -- fires once at t=0
    rows = gather_emitter_results(sim)[("emitter",)]
    row = next(r for r in rows if r.get("n_cells"))
    return {
        "n_cells": int(row["n_cells"]),
        "solution": np.asarray(row["solution"], dtype=float),
        "solution_max": float(row["solution_max"]),
    }


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
        "n_steps": len(GEOMETRIES),
        "emitter": "ram",
        "origin": "canonical_run",
        "study_slug": STUDY_SLUG,
        "investigation_slug": INVESTIGATION_SLUG,
        "params": {"resolution": RESOLUTION, "geometries": list(GEOMETRIES)},
    })

    try:
        core = build_core()
        results = {}
        for geometry in GEOMETRIES:
            results[geometry] = _run_one(core, geometry)

        n_cells_by_geom = {g: r["n_cells"] for g, r in results.items()}
        min_n_cells = min(n_cells_by_geom.values())

        solution_max_by_geom = {g: r["solution_max"] for g, r in results.items()}
        min_solution_max = min(solution_max_by_geom.values())
        max_solution_max = max(solution_max_by_geom.values())

        all_finite = all(np.all(np.isfinite(r["solution"])) for r in results.values())
        all_nonnegative = all(np.min(r["solution"]) > -1e-8 for r in results.values())

        mesh_imports_ok = min_n_cells > MIN_N_CELLS
        solution_finite_bounded_nontrivial = (
            all_finite
            and all_nonnegative
            and min_solution_max > SOLUTION_MAX_FLOOR
        )
        solution_stays_bounded = max_solution_max < SOLUTION_MAX_CEILING

        # --- viz: obstacle 2D heatmap + 3D relief mesh ---
        obstacle_domain, obstacle_V = fem_gmsh.build_gmsh_mesh("obstacle", RESOLUTION)
        obstacle_solution = fem_gmsh.solve_poisson_gmsh(obstacle_domain, obstacle_V, source_value=1.0)
        coords2 = fem_gmsh.node_coords(obstacle_V)
        cells = fem_gmsh.triangle_cells(obstacle_domain)

        viz_dir = STUDY_DIR / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)

        heatmap_html = viz.field_heatmap_html(
            coords2, obstacle_solution,
            f"Poisson solution on obstacle domain (gmsh, resolution={RESOLUTION})",
        )
        (viz_dir / "field_obstacle.html").write_text(heatmap_html)

        # Lift the 2D field into 3D: z = value (scaled for a visible
        # relief), colored by the same value, on the imported mesh's real
        # triangle connectivity.
        z_scale = 3.0
        coords3 = np.column_stack([coords2, obstacle_solution * z_scale])
        mesh3d_html = viz.mesh3d_html(
            coords3, cells, obstacle_solution,
            title=f"Poisson solution relief (obstacle domain, gmsh, resolution={RESOLUTION})",
        )
        (viz_dir / "mesh3d.html").write_text(mesh3d_html)
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
        "n_steps": len(GEOMETRIES),
        "status": "completed",
    })

    # --- report: PASS/FAIL mirrors this study's declared behavior_tests
    # exactly (mesh-imports-with-positive-cells,
    # solution-finite-bounded-nontrivial, solution-stays-bounded) -- no
    # additional undeclared conditions.
    for g in GEOMETRIES:
        print(
            f"[complex-geometry] geometry={g} resolution={RESOLUTION}: "
            f"n_cells={n_cells_by_geom[g]} solution_max={solution_max_by_geom[g]:.5f}"
        )
    print(
        f"[complex-geometry] min n_cells across geometries={min_n_cells} "
        f"-> {'PASS' if mesh_imports_ok else 'FAIL'} (mesh-imports-with-positive-cells > {MIN_N_CELLS})"
    )
    print(
        f"[complex-geometry] all finite={all_finite}, all non-negative={all_nonnegative}, "
        f"min solution_max={min_solution_max:.5f} "
        f"-> {'PASS' if solution_finite_bounded_nontrivial else 'FAIL'} "
        f"(solution-finite-bounded-nontrivial: finite & non-negative & max solution_max > {SOLUTION_MAX_FLOOR:.0e})"
    )
    print(
        f"[complex-geometry] max solution_max across geometries={max_solution_max:.5f} "
        f"-> {'PASS' if solution_stays_bounded else 'FAIL'} (solution-stays-bounded < {SOLUTION_MAX_CEILING})"
    )
    print(f"[complex-geometry] viz written: {viz_dir / 'field_obstacle.html'}")
    print(f"[complex-geometry] viz written: {viz_dir / 'mesh3d.html'}")
    print(f"[complex-geometry] run recorded in {WORKSPACE_ROOT / '.pbg' / 'runs.jsonl'}")

    all_pass = mesh_imports_ok and solution_finite_bounded_nontrivial and solution_stays_bounded

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
