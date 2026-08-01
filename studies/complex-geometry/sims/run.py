#!/usr/bin/env python3
"""Canonical run for the ``complex-geometry`` study -- Stokes flow through a
2D porous lattice, with a computed effective permeability.

Builds the ``porous_lattice`` composite (a single ``PorousFlowStep`` -- real
gmsh pillar-array geometry -> dolfinx Taylor-Hood MIXED import -> steady
Stokes solve, see ``viva_fenics/processes/flow.py``'s module comment above
``PorousFlowStep`` for the physics/BC derivation) at the baseline porosity
plus 3 porosity variants (sweeping ``pillar_radius`` at fixed nx=ny=4
lattice density):

1. For each porosity, solves creeping (Stokes) flow through the pillar
   lattice, driven by a prescribed pressure drop, and checks the velocity
   field is (approximately) divergence-free and exactly no-slip on every
   pillar + wall boundary.
2. Computes the effective (Darcy) permeability k_eff = mu * <u_x> * L /
   delta_p at each porosity, and checks k_eff DECREASES monotonically as
   porosity decreases (denser pillar packing -> lower permeability) -- the
   Kozeny-Carman-type physical trend this study exists to demonstrate.
3. Renders (a) the velocity-magnitude field + streamlines threading through
   the baseline pillar array, and (b) a k_eff-vs-porosity chart with a
   fitted Kozeny-Carman-type reference curve overlaid for shape comparison.
4. Prints PASS/FAIL mirroring the study's declared behavior_tests exactly,
   plus the achieved k_eff(porosity) values.

Standalone; run from the workspace root::

    pixi run python studies/complex-geometry/sims/run.py
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

from viva_fenics import fem_gmsh, viz
from viva_fenics.core import build_core
from viva_fenics.composites.flow import porous_lattice
from viva_fenics.processes.flow import porous_velocity_coords
from vivarium_workbench.lib.run_log import append_run_event

SPEC_ID = "viva_fenics.composites.flow.porous_lattice"
STUDY_SLUG = "complex-geometry"
INVESTIGATION_SLUG = "fenics-showcase"

# Fixed lattice density (nx x ny pillar grid) and mesh grading across every
# porosity in this sweep -- only pillar_radius varies, so porosity is the
# ONLY thing changing between runs (a clean single-parameter sweep).
NX, NY = 4, 4
H_PILLAR = 0.015
H_FAR = 0.05
MU = 1.0
PRESSURE_DROP = 1.0

# Porosity sweep: pillar_radius from sparse (large pores, high porosity) to
# dense (small pores, low porosity) -- all safely below the nx=ny=4
# lattice's overlap ceiling (spacing=0.25, so pillar_radius < 0.125).
VARIANTS = [
    ("sparse", 0.04),
    ("medium", 0.06),
    ("baseline", 0.08),
    ("dense", 0.10),
]

# Behavior-test thresholds (see study.yaml's expected_behavior/behavior_tests
# for the same numbers with rationale). Calibrated around observed values
# from a development spike (divergence_mean ~1e-4-3e-4, noslip_max_speed
# exactly 0.0 -- Dirichlet dof elimination is exact regardless of mesh
# resolution) -- generous headroom on both, tight enough to catch a
# genuinely broken BC/assembly.
DIVERGENCE_THRESHOLD = 1e-2
NOSLIP_THRESHOLD = 1e-8
MIN_N_CELLS = 20
K_EFF_FLOOR = 1e-8


def _run_one(core, pillar_radius):
    doc = porous_lattice(
        core, nx=NX, ny=NY, pillar_radius=pillar_radius,
        h_pillar=H_PILLAR, h_far=H_FAR, mu=MU, pressure_drop=PRESSURE_DROP,
    )
    sim = Composite({"state": doc}, core=core)
    sim.run(0)  # PorousFlowStep is stateless -- fires once at t=0
    rows = gather_emitter_results(sim)[("emitter",)]
    row = next(r for r in rows if r.get("n_cells"))
    return {
        "n_cells": int(row["n_cells"]),
        "porosity": float(row["porosity"]),
        "mean_ux": float(row["mean_ux"]),
        "k_eff": float(row["k_eff"]),
        "divergence_mean": float(row["divergence_mean"]),
        "noslip_max_speed": float(row["noslip_max_speed"]),
        "velocity_x": np.asarray(row["velocity_x"], dtype=float),
        "velocity_y": np.asarray(row["velocity_y"], dtype=float),
    }


def _subsample_for_quiver(coords, vx, vy, speed, n_bins=36):
    """Thin a dense P2 velocity dof cloud down to ~1 real node per coarse
    spatial bin, for a LEGIBLE quiver plot -- the raw P2 velocity space at
    production mesh resolution has 10000+ dofs (every vertex AND every edge
    midpoint), which renders as an unreadable solid mass of overlapping
    arrows. Every returned point is a REAL solved mesh node/value (nearest
    node to each occupied bin's center) -- this is spatial selection, not
    interpolation/smoothing, so it introduces no fabricated data.
    """
    coords = np.asarray(coords, dtype=float)
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    bin_w = (xmax - xmin) / n_bins or 1.0
    bin_h = (ymax - ymin) / n_bins or 1.0
    bin_ix = np.clip(((coords[:, 0] - xmin) / bin_w).astype(int), 0, n_bins - 1)
    bin_iy = np.clip(((coords[:, 1] - ymin) / bin_h).astype(int), 0, n_bins - 1)
    bin_key = bin_ix * n_bins + bin_iy

    bin_centers_x = xmin + (bin_ix + 0.5) * bin_w
    bin_centers_y = ymin + (bin_iy + 0.5) * bin_h
    dist_to_center = np.hypot(coords[:, 0] - bin_centers_x, coords[:, 1] - bin_centers_y)

    chosen = {}
    for i in range(len(coords)):
        key = bin_key[i]
        if key not in chosen or dist_to_center[i] < chosen[key][0]:
            chosen[key] = (dist_to_center[i], i)
    idx = np.array([i for _, i in chosen.values()])
    return coords[idx], vx[idx], vy[idx], speed[idx]


def _kozeny_carman_fit(porosities, k_effs):
    """Fit a Kozeny-Carman-SHAPED curve k(phi) = C * phi^3 / (1-phi)^2 to
    the observed (porosity, k_eff) points by least-squares fitting ONLY the
    normalization constant C in log-space (the canonical exponents 3/-2 are
    fixed, not fitted) -- a shape/trend comparison, not an independent
    scaling-law validation: with only ``len(VARIANTS)`` points and one free
    parameter, this cannot distinguish the KC exponents from a nearby
    alternative power law. See run.py's printed summary for the honest
    framing.

    Returns:
        (log_c, kc_curve_fn) tuple -- ``kc_curve_fn(phi)`` evaluates the
        fitted curve at arbitrary porosity values.
    """
    phi = np.asarray(porosities, dtype=float)
    k = np.asarray(k_effs, dtype=float)
    shape = 3.0 * np.log(phi) - 2.0 * np.log(1.0 - phi)
    log_c = float(np.mean(np.log(k) - shape))

    def kc_curve_fn(phi_query):
        phi_query = np.asarray(phi_query, dtype=float)
        return np.exp(log_c) * phi_query**3 / (1.0 - phi_query) ** 2

    return log_c, kc_curve_fn


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
        "n_steps": len(VARIANTS),
        "emitter": "ram",
        "origin": "canonical_run",
        "study_slug": STUDY_SLUG,
        "investigation_slug": INVESTIGATION_SLUG,
        "params": {
            "nx": NX, "ny": NY, "h_pillar": H_PILLAR, "h_far": H_FAR,
            "mu": MU, "pressure_drop": PRESSURE_DROP,
            "pillar_radii": [r for _, r in VARIANTS],
        },
    })

    try:
        core = build_core()
        solve_t0 = time.time()
        results = {}
        for name, pillar_radius in VARIANTS:
            results[name] = _run_one(core, pillar_radius)
            r = results[name]
            print(
                f"[complex-geometry] variant={name} pillar_radius={pillar_radius} "
                f"porosity={r['porosity']:.4f} n_cells={r['n_cells']} "
                f"mean_ux={r['mean_ux']:.6f} k_eff={r['k_eff']:.6e} "
                f"div_mean={r['divergence_mean']:.3e} noslip_max={r['noslip_max_speed']:.3e}",
                flush=True,
            )
        solve_wall_time = time.time() - solve_t0

        if len(results) < 2:
            print("[complex-geometry] ERROR: fewer than 2 variants ran", file=sys.stderr)
            append_run_event(WORKSPACE_ROOT, {
                "run_id": run_id, "event": "completed",
                "completed_at": time.time(), "n_steps": len(results), "status": "failed",
            })
            return 1

        # Sort by porosity DESCENDING -- the order the permeability-trend
        # check (and the KC-fit curve) expects.
        ordered = sorted(results.values(), key=lambda r: r["porosity"], reverse=True)
        porosities = [r["porosity"] for r in ordered]
        k_effs = [r["k_eff"] for r in ordered]
        n_cells_list = [r["n_cells"] for r in ordered]
        divergence_list = [r["divergence_mean"] for r in ordered]
        noslip_list = [r["noslip_max_speed"] for r in ordered]

        min_n_cells = min(n_cells_list)
        max_divergence = max(divergence_list)
        max_noslip = max(noslip_list)
        min_k_eff = min(k_effs)
        all_k_eff_finite = all(np.isfinite(k) for k in k_effs)

        consecutive_ratios = [k_effs[i] / k_effs[i + 1] for i in range(len(k_effs) - 1)]
        min_consecutive_ratio = min(consecutive_ratios)

        mesh_imports_ok = min_n_cells > MIN_N_CELLS
        divergence_free_and_noslip = (
            max_divergence < DIVERGENCE_THRESHOLD and max_noslip < NOSLIP_THRESHOLD
        )
        noslip_ok = max_noslip < NOSLIP_THRESHOLD
        permeability_finite_positive = all_k_eff_finite and min_k_eff > K_EFF_FLOOR
        permeability_decreases_with_porosity = min_consecutive_ratio > 1.0

        # --- Kozeny-Carman-shaped fit (shape comparison only -- see
        # _kozeny_carman_fit's docstring for the honest framing) ---
        _log_c, kc_curve_fn = _kozeny_carman_fit(porosities, k_effs)
        kc_curve = kc_curve_fn(np.asarray(porosities))
        kc_residuals = np.abs(np.log(k_effs) - np.log(kc_curve))
        max_kc_log_residual = float(np.max(kc_residuals))

        # --- viz: baseline flow field (velocity magnitude + streamlines) ---
        viz_dir = STUDY_DIR / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)

        baseline = results["baseline"]
        coords = porous_velocity_coords(NX, NY, 0.08, H_PILLAR, H_FAR)
        vx, vy = baseline["velocity_x"], baseline["velocity_y"]
        speed = np.hypot(vx, vy)
        # Thin the dense P2 dof cloud (15000+ nodes) to a legible arrow
        # density -- see _subsample_for_quiver's docstring: every arrow is
        # still a REAL solved node/value, just spatially subsampled.
        coords_q, vx_q, vy_q, speed_q = _subsample_for_quiver(coords, vx, vy, speed)
        flow_field_html = viz.quiver_streamlines_html(
            coords_q, vx_q, vy_q, speed_q,
            f"Stokes flow through porous pillar lattice (porosity={baseline['porosity']:.3f}, "
            f"{NX}x{NY} pillars)",
        )
        (viz_dir / "flow_field.html").write_text(flow_field_html)

        speed_heatmap_html = viz.field_heatmap_html(
            coords, speed,
            f"Velocity magnitude through the pillar lattice (porosity={baseline['porosity']:.3f})",
        )
        (viz_dir / "velocity_magnitude.html").write_text(speed_heatmap_html)

        # --- viz: k_eff vs porosity, with KC-shaped reference curve ---
        phi_dense = np.linspace(min(porosities) * 0.9, max(porosities) * 1.02, 60)
        kc_dense = kc_curve_fn(phi_dense)
        permeability_html = viz.profile_with_fit_html(
            porosities, k_effs, kc_curve,
            title="Effective permeability k_eff vs porosity (Stokes flow, pillar lattice)",
            x_label="porosity (phi)",
            y_label="k_eff",
            measured_label="k_eff (Stokes solve)",
            fit_label="Kozeny-Carman-shaped fit (phi^3/(1-phi)^2)",
            log_y=True,
            annotation=(
                f"k_eff falls {porosities[0]:.2f}->{porosities[-1]:.2f} porosity: "
                f"{k_effs[0]:.2e} -> {k_effs[-1]:.2e} "
                f"({k_effs[0] / k_effs[-1]:.1f}x drop)"
            ),
        )
        (viz_dir / "permeability_vs_porosity.html").write_text(permeability_html)
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
        "n_steps": len(VARIANTS),
        "status": "completed",
    })

    # --- report: PASS/FAIL mirrors this study's declared behavior_tests
    # exactly (mesh-imports-with-tagged-boundaries,
    # stokes-flow-divergence-free-and-noslip, noslip-satisfied-on-pillars,
    # permeability-decreases-with-porosity) -- no additional undeclared
    # conditions.
    print(f"[complex-geometry] solve wall-time: {solve_wall_time:.2f}s for {len(VARIANTS)} variants")
    print(
        f"[complex-geometry] porosity sweep (descending): "
        + ", ".join(f"{p:.4f}" for p in porosities)
    )
    print(
        f"[complex-geometry] k_eff sweep (matching order): "
        + ", ".join(f"{k:.4e}" for k in k_effs)
    )
    print(
        f"[complex-geometry] Kozeny-Carman-shaped fit (C fitted, exponents 3/-2 fixed): "
        f"max log-residual={max_kc_log_residual:.3f} (qualitative shape comparison only, "
        f"n={len(VARIANTS)} points -- NOT an independent validation of the KC exponents)"
    )
    print(
        f"[complex-geometry] min n_cells across variants={min_n_cells} "
        f"-> {'PASS' if mesh_imports_ok else 'FAIL'} (mesh-imports-with-tagged-boundaries > {MIN_N_CELLS})"
    )
    print(
        f"[complex-geometry] max divergence_mean={max_divergence:.3e}, max noslip_max_speed={max_noslip:.3e} "
        f"-> {'PASS' if divergence_free_and_noslip else 'FAIL'} "
        f"(stokes-flow-divergence-free-and-noslip: div < {DIVERGENCE_THRESHOLD:.0e} and noslip < {NOSLIP_THRESHOLD:.0e})"
    )
    print(
        f"[complex-geometry] max noslip_max_speed={max_noslip:.3e} "
        f"-> {'PASS' if noslip_ok else 'FAIL'} (noslip-satisfied-on-pillars < {NOSLIP_THRESHOLD:.0e})"
    )
    print(
        f"[complex-geometry] min consecutive k_eff ratio={min_consecutive_ratio:.4f}, "
        f"min k_eff={min_k_eff:.4e}, all finite={all_k_eff_finite} "
        f"-> {'PASS' if permeability_decreases_with_porosity and permeability_finite_positive else 'FAIL'} "
        f"(permeability-decreases-with-porosity: min ratio > 1.0, and k_eff finite & > {K_EFF_FLOOR:.0e})"
    )
    print(
        f"[complex-geometry] viz written: {viz_dir / 'flow_field.html'}, "
        f"{viz_dir / 'velocity_magnitude.html'}, {viz_dir / 'permeability_vs_porosity.html'}"
    )
    print(f"[complex-geometry] run recorded in {WORKSPACE_ROOT / '.pbg' / 'runs.jsonl'}")

    all_pass = (
        mesh_imports_ok
        and divergence_free_and_noslip
        and noslip_ok
        and permeability_decreases_with_porosity
        and permeability_finite_positive
    )

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
