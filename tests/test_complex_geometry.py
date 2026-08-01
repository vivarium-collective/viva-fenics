"""Behavior tests for the porous-lattice Stokes solve (real gmsh pillar-array
geometry -> dolfinx Taylor-Hood mixed import -> steady Stokes solve -> Darcy
effective permeability -- see viva_fenics/fem_gmsh.py's porous-lattice
section and viva_fenics/processes/flow.py's PorousFlowStep module docstring).

Small lattice (nx=ny=2) + coarse mesh sizes for pytest speed; see
studies/complex-geometry/sims/run.py for the full production-resolution run.
"""

import numpy as np
from process_bigraph import Composite, allocate_core, gather_emitter_results

from viva_fenics import fem_gmsh
from viva_fenics.processes.flow import (
    PorousFlowStep,
    solve_porous_stokes,
    mean_velocity_x,
    noslip_max_speed,
    divergence_stats,
)

NX, NY = 2, 2
H_PILLAR = 0.03
H_FAR = 0.08
MU = 1.0
PRESSURE_DROP = 1.0

# Three porosities, small lattice, all safely below the nx=ny=2 overlap
# ceiling (spacing=0.5, so pillar_radius must stay < 0.25).
RADII = (0.08, 0.14, 0.18)


# ---------------------------------------------------------------------------
# (a) mesh imports with tagged pillar boundaries + correct porosity
# ---------------------------------------------------------------------------

def test_porous_lattice_mesh_has_tagged_boundaries():
    """The gmsh-generated porous-lattice mesh must import with a non-trivial
    cell count and every one of the 4 tagged boundary facet groups
    (inflow/outflow/walls/pillars) populated -- if the bounding-box
    classification in fem_gmsh._build_porous_lattice_model ever
    misclassifies a boundary curve, one of these groups silently comes back
    empty and the no-slip/pressure-load BCs would apply to nothing.
    """
    domain, facet_tags, markers, porosity = fem_gmsh.build_porous_lattice_mesh(
        NX, NY, 0.14, H_PILLAR, H_FAR
    )
    assert fem_gmsh.n_cells(domain) > 0
    for name in ("inflow", "outflow", "walls", "pillars"):
        n_facets = len(facet_tags.find(markers[name]))
        assert n_facets > 0, f"boundary group {name!r} has no tagged facets"
    # nx*ny=4 pillars -> 4 disjoint circular boundary loops tagged "pillars".
    assert len(facet_tags.find(markers["pillars"])) > len(facet_tags.find(markers["inflow"]))


def test_porous_lattice_porosity_matches_analytic_estimate():
    """The FEM-assembled (real meshed-domain) porosity must land close to
    the closed-form geometry estimate (``lattice_porosity``) -- confirms
    the boolean cut actually removed the right amount of area, not some
    other (e.g. doubled/halved) fraction.
    """
    for r in RADII:
        domain, facet_tags, markers, porosity = fem_gmsh.build_porous_lattice_mesh(
            NX, NY, r, H_PILLAR, H_FAR
        )
        analytic = fem_gmsh.lattice_porosity(NX, NY, r)
        assert 0.0 < porosity < 1.0
        assert abs(porosity - analytic) < 0.02, (
            f"r={r}: FEM porosity {porosity:.4f} vs analytic {analytic:.4f} "
            "differ by more than the expected circle-triangulation error"
        )


def test_porous_lattice_rejects_overlapping_pillars():
    """pillar_radius >= half the lattice spacing would overlap neighboring
    pillars (or the channel wall) -- must raise, not silently build a
    corrupted geometry."""
    import pytest
    with pytest.raises(ValueError):
        fem_gmsh.build_porous_lattice_mesh(NX, NY, 0.26, H_PILLAR, H_FAR)


# ---------------------------------------------------------------------------
# (b) Stokes solve: divergence-free + no-slip on pillars
# ---------------------------------------------------------------------------

def test_stokes_solve_is_finite_and_divergence_free():
    """A real Taylor-Hood mixed Stokes solve must produce a finite, bounded
    velocity/pressure field with SMALL mean |div(u)| -- Taylor-Hood is an
    LBB-stable pairing, so (unlike the IPCS operator-splitting processes
    elsewhere in this module) this should be much closer to exactly
    divergence-free, not just approximately so.
    """
    for r in RADII:
        domain, facet_tags, markers, uh, ph, porosity = solve_porous_stokes(
            NX, NY, r, H_PILLAR, H_FAR, MU, PRESSURE_DROP
        )
        assert np.all(np.isfinite(uh.x.array))
        assert np.all(np.isfinite(ph.x.array))
        mean_div, max_div = divergence_stats(domain, uh.function_space, uh.x.array)
        assert mean_div < 1e-2, f"r={r}: mean|div(u)|={mean_div} unexpectedly large for Taylor-Hood"


def test_stokes_solve_satisfies_noslip_on_pillars_and_walls():
    """Every no-slip (wall + pillar) boundary dof must have (near-)exactly
    zero velocity -- confirms the Dirichlet BC actually landed on the real
    tagged facets, not silently skipped."""
    for r in RADII:
        domain, facet_tags, markers, uh, ph, porosity = solve_porous_stokes(
            NX, NY, r, H_PILLAR, H_FAR, MU, PRESSURE_DROP
        )
        max_speed = noslip_max_speed(domain, facet_tags, markers, uh)
        assert max_speed < 1e-10, f"r={r}: no-slip violated, max|u|={max_speed}"


def test_flow_is_driven_in_the_pressure_gradient_direction():
    """The pressure-drop-driven flow's volume-averaged x-velocity must be
    positive (flow moves from high to low pressure, i.e. +x, matching the
    inflow-high/outflow-low convention) -- not zero, not backwards."""
    domain, facet_tags, markers, uh, ph, porosity = solve_porous_stokes(
        NX, NY, 0.14, H_PILLAR, H_FAR, MU, PRESSURE_DROP
    )
    mean_ux = mean_velocity_x(domain, uh)
    assert mean_ux > 0.0


# ---------------------------------------------------------------------------
# (c) k_eff: finite, positive, and DECREASES as porosity decreases
# ---------------------------------------------------------------------------

def test_k_eff_finite_positive_and_decreasing_with_porosity():
    """Effective permeability (Darcy's law solved for k_eff) must be
    finite, strictly positive, and DECREASE monotonically as porosity
    decreases (denser pillar packing -> lower permeability) -- the
    Kozeny-Carman-type physical trend this study exists to demonstrate.
    Real Stokes solves at 3 porosities on a small test-scale lattice, not a
    fabricated/asserted trend.
    """
    results = []
    for r in RADII:
        domain, facet_tags, markers, uh, ph, porosity = solve_porous_stokes(
            NX, NY, r, H_PILLAR, H_FAR, MU, PRESSURE_DROP
        )
        mean_ux = mean_velocity_x(domain, uh)
        k_eff = MU * mean_ux * fem_gmsh.POROUS_DOMAIN_LENGTH / PRESSURE_DROP
        assert np.isfinite(k_eff)
        assert k_eff > 0.0
        results.append((porosity, k_eff))

    # RADII is increasing (smaller pillars -> larger porosity omitted; here
    # increasing radius -> decreasing porosity), so results are already in
    # decreasing-porosity order; k_eff must decrease in lockstep.
    results.sort(key=lambda pk: pk[0], reverse=True)  # sort by porosity, descending
    porosities = [p for p, _ in results]
    k_effs = [k for _, k in results]
    assert porosities == sorted(porosities, reverse=True)
    for i in range(len(k_effs) - 1):
        assert k_effs[i] > k_effs[i + 1], (
            f"k_eff should decrease as porosity decreases: {results}"
        )


# ---------------------------------------------------------------------------
# PorousFlowStep (the process-bigraph wrapper)
# ---------------------------------------------------------------------------

def test_porous_flow_step_update():
    core = allocate_core()
    step = PorousFlowStep(
        config={
            "nx": NX, "ny": NY, "pillar_radius": 0.14,
            "h_pillar": H_PILLAR, "h_far": H_FAR,
            "mu": MU, "pressure_drop": PRESSURE_DROP,
        },
        core=core,
    )
    out = step.update({})
    assert out["n_cells"] > 0
    assert len(out["velocity_x"]) > 0
    assert len(out["velocity_x"]) == len(out["velocity_y"])
    assert np.isfinite(out["k_eff"]) and out["k_eff"] > 0.0
    assert 0.0 < out["porosity"] < 1.0
    assert out["noslip_max_speed"] < 1e-10
    assert out["divergence_mean"] < 1e-2


def test_porous_lattice_generator_registered():
    from viva_superpowers.composite_generator import _REGISTRY
    assert any(e.endswith(".porous_lattice") for e in _REGISTRY)


def test_porous_lattice_composite_runs_and_emits():
    from viva_fenics.core import build_core
    from viva_fenics.composites.flow import porous_lattice

    core = build_core()
    document = porous_lattice(
        core=core, nx=NX, ny=NY, pillar_radius=0.14,
        h_pillar=H_PILLAR, h_far=H_FAR, mu=MU, pressure_drop=PRESSURE_DROP,
    )
    sim = Composite({"state": document}, core=core)
    sim.run(0)  # Step-only composite: a single triggered update is enough

    rows = gather_emitter_results(sim)[("emitter",)]
    assert len(rows) >= 1
    last = rows[-1]
    assert last["n_cells"] > 0
    assert len(last["velocity_x"]) > 0
    assert np.isfinite(last["k_eff"]) and last["k_eff"] > 0.0
