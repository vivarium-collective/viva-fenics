"""Multisolute 3-D pore transport: the geometry, the solver generalisation to
3-D, the per-species Soret + adsorption + degradation physics, the convergence
envelope, and the composite. All drive the REAL dolfinx solve -- skipped
(via conftest) when the backend is absent, never mocked.
"""
import numpy as np
import pytest

from dolfinx import fem

from viva_fenics import fem as F
from viva_fenics.fem_pore3d import build_pore3d_mesh, PORE3D_MARKERS
from viva_fenics.fem_multisolute import (
    adsorb_degrade,
    colocalization,
    enrichment,
    multisolute_pore_step,
)
from viva_fenics.fem_convergence import (
    mesh_convergence,
    run_pore_enrichment,
    self_convergence,
)


# --- geometry ---------------------------------------------------------------

def test_pore3d_mesh_is_3d_and_tagged():
    domain, ftags, markers = build_pore3d_mesh(h_throat=0.06, h_chamber=0.12)
    assert domain.geometry.dim == 3
    assert markers == PORE3D_MARKERS
    # every named boundary resolves to a nonempty facet set
    for name in ("inlet", "outlet", "wall"):
        assert len(ftags.find(markers[name])) > 0
    # the wall carries far more facets than either cap (it is the long lateral)
    assert len(ftags.find(markers["wall"])) > len(ftags.find(markers["inlet"]))


def test_pore3d_rejects_throat_wider_than_chamber():
    with pytest.raises(ValueError):
        build_pore3d_mesh(chamber_radius=0.2, throat_radius=0.3)


# --- solver generalisation to 3-D ------------------------------------------

def test_advection_diffusion_accepts_3_component_velocity():
    """The gdim-general constant-velocity path drives the 3-D pore (regression
    for the old hardcoded (wx, wy))."""
    domain, ftags, markers = build_pore3d_mesh(h_throat=0.08, h_chamber=0.14)
    V = fem.functionspace(domain, ("Lagrange", 1))
    n = V.dofmap.index_map.size_local
    out = F.advection_diffusion_step(
        domain, V, np.zeros(n), np.ones(n) * 0.1, dt=0.05, D=0.05,
        velocity=(0.0, 0.0, 0.3),
    )
    assert out.shape == (n,)
    assert np.all(np.isfinite(out))


def test_constant_velocity_wrong_dimension_raises():
    domain, _, _ = build_pore3d_mesh(h_throat=0.1, h_chamber=0.16)
    V = fem.functionspace(domain, ("Lagrange", 1))
    n = V.dofmap.index_map.size_local
    with pytest.raises(ValueError):
        F.advection_diffusion_step(domain, V, np.zeros(n), np.zeros(n),
                                   dt=0.05, D=0.05, velocity=(1.0, 0.0))  # 2-D on 3-D


# --- analytic limit ---------------------------------------------------------

def test_no_soret_no_flow_relaxes_to_feed():
    """Pure diffusion with a Dirichlet inlet feed and zero-flux walls/outlet
    relaxes to the uniform feed concentration everywhere -- the trivial analytic
    steady state (a constant satisfies Laplace + these BCs). Enrichment -> 1 (no
    concentration). A mild throat is used so diffusion equilibrates through the
    constriction in a reasonable number of steps; it is still a real 3-D solve."""
    domain, ftags, markers = build_pore3d_mesh(
        chamber_radius=0.4, throat_radius=0.3, h_throat=0.08, h_chamber=0.12)
    V = fem.functionspace(domain, ("Lagrange", 1))
    n = V.dofmap.index_map.size_local
    species = [{"name": "t", "dissolved": np.zeros(n), "adsorbed": np.zeros(n),
                "D": 0.3, "soret": 0.0, "inlet": 1.0}]
    prev_min = -1.0
    for _ in range(250):
        species = multisolute_pore_step(domain, V, ftags, markers, species,
                                        dt=0.1, velocity=None, temperature_array=None)
        cur_min = float(species[0]["dissolved"].min())
        assert cur_min >= prev_min - 1e-9   # monotone fill, no undershoot
        prev_min = cur_min
    field = species[0]["dissolved"]
    assert field.max() <= 1.0 + 1e-6        # no overshoot above the feed
    assert np.allclose(field, 1.0, atol=2e-2)  # reached the uniform steady state
    assert enrichment(domain, V, ftags, markers, field) == pytest.approx(1.0, abs=2e-2)


# --- multisolute physics ----------------------------------------------------

def _run_mixture(species, n_steps=80, peclet_velocity=0.25, delta_T=30.0):
    domain, ftags, markers = build_pore3d_mesh(h_throat=0.06, h_chamber=0.12)
    V = fem.functionspace(domain, ("Lagrange", 1))
    n = V.dofmap.index_map.size_local
    z = V.tabulate_dof_coordinates()[:, 2]
    T = 330.0 - delta_T * z
    specs = [dict(s, dissolved=np.zeros(n), adsorbed=np.zeros(n)) for s in species]
    for _ in range(n_steps):
        specs = multisolute_pore_step(domain, V, ftags, markers, specs, dt=0.02,
                                      velocity=(0.0, 0.0, peclet_velocity),
                                      temperature_array=T)
    return domain, V, ftags, markers, specs


def test_soret_enrichment_orders_by_coefficient():
    """Higher Soret coefficient -> stronger cold-side concentration."""
    _, V, ftags, markers, specs = _run_mixture([
        {"name": "a", "D": 0.05, "soret": 0.03, "inlet": 1.0},
        {"name": "b", "D": 0.05, "soret": 0.09, "inlet": 1.0},
        {"name": "c", "D": 0.05, "soret": 0.15, "inlet": 1.0},
    ])
    domain = V.mesh
    e = [enrichment(domain, V, ftags, markers, s["dissolved"]) for s in specs]
    assert e[2] > e[1] > e[0] > 1.0  # all concentrate; strongest Soret wins


def test_no_soret_species_does_not_concentrate():
    _, V, ftags, markers, specs = _run_mixture([
        {"name": "inert", "D": 0.05, "soret": 0.0, "inlet": 1.0},
    ])
    domain = V.mesh
    assert enrichment(domain, V, ftags, markers, specs[0]["dissolved"]) == pytest.approx(1.0, abs=5e-2)


def test_adsorption_conserves_dissolved_plus_adsorbed():
    d = np.array([1.0, 2.0, 0.5, 3.0])
    a = np.array([0.1, 0.0, 0.2, 0.0])
    d2, a2 = adsorb_degrade(d, a, dt=0.7, k_ads=0.9, k_des=0.3, k_deg=0.0)
    assert np.allclose(d + a, d2 + a2)          # mass conserved
    assert np.all(a2 >= a - 1e-12)              # net adsorption from zero-ish a
    # equilibrium split a_eq = total * k_ads/(k_ads+k_des)
    total = d + a
    assert np.all(a2 <= total * 0.9 / 1.2 + 1e-9)


def test_degradation_is_exact_first_order():
    d = np.array([1.0, 2.0, 0.5])
    a = np.zeros(3)
    d2, _ = adsorb_degrade(d, a, dt=1.3, k_ads=0.0, k_des=0.0, k_deg=0.4)
    assert np.allclose(d2, d * np.exp(-0.4 * 1.3))


def test_degradation_lowers_enrichment():
    """A degrading solute concentrates less than an identical stable one."""
    stable = [{"name": "s", "D": 0.05, "soret": 0.12, "inlet": 1.0}]
    labile = [{"name": "l", "D": 0.05, "soret": 0.12, "inlet": 1.0, "k_deg": 0.6}]
    _, V, ftags, markers, ss = _run_mixture(stable)
    _, V2, ftags2, markers2, ls = _run_mixture(labile)
    e_stable = enrichment(V.mesh, V, ftags, markers, ss[0]["dissolved"])
    e_labile = enrichment(V2.mesh, V2, ftags2, markers2, ls[0]["dissolved"])
    assert e_labile < e_stable


# --- co-localization metric -------------------------------------------------

def test_colocalization_identical_and_anti():
    x = np.linspace(0, 1, 50)
    assert colocalization(x, x) == pytest.approx(1.0, abs=1e-9)
    assert colocalization(x, -x) == pytest.approx(-1.0, abs=1e-9)
    assert np.isnan(colocalization(np.ones(10), x))  # uniform field -> NaN


# --- convergence envelope ---------------------------------------------------

def test_self_convergence_flags_convergence():
    assert self_convergence([1.0, 1.1, 1.102], rtol=0.05)["converged"]
    assert not self_convergence([1.0, 1.4, 1.9], rtol=0.05)["converged"]


def test_low_peclet_mesh_converges():
    """At a modest Peclet the enrichment readout is mesh-converged (the
    trustworthy regime)."""
    mc = mesh_convergence(peclet=4.0, h_throats=(0.09, 0.06, 0.04),
                          n_steps=50, rtol=0.08)
    assert mc["converged"]
    assert all(np.isfinite(lv["enrichment"]) for lv in mc["levels"])


def test_run_pore_enrichment_is_finite_and_above_one():
    r = run_pore_enrichment(peclet=6.0, soret=0.12, h_throat=0.06, n_steps=60)
    assert np.isfinite(r["enrichment"]) and r["enrichment"] > 1.0
    assert r["dofs"] > 0 and r["n_cells"] > 0


# --- composite via build_core ----------------------------------------------

def test_multisolute_pore_composite_runs():
    from process_bigraph import Composite, gather_emitter_results
    from viva_fenics.core import build_core
    from viva_fenics.composites.pore3d import multisolute_pore

    core = build_core()
    sim = Composite({"state": multisolute_pore(core, peclet=6.0, n_steps=50)}, core=core)
    sim.run(1.0)
    res = gather_emitter_results(sim)[("emitter",)][-1]
    assert res["n_species"] == 3
    enr = [float(x) for x in res["enrichment"]]
    assert len(enr) == 3 and all(np.isfinite(enr))
    assert enr[2] > enr[0]  # slow_strong (most Soret) > fast_weak
    assert np.isfinite(res["colocalization_min"])
