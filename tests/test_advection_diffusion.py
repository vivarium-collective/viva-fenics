"""Real-dolfinx tests for the advection + thermophoretic-drift extension to the
backward-Euler reactive-transport weak form (viva-genesis full modeling plan
section 4.5). Every solve below runs the genuine FEM assembly + PETSc LU solve
in ``viva_fenics.fem.advection_diffusion_step`` -- no mock, no analytic stand-in
(the analytic forms appear only as the error reference a real solve is checked
against).
"""

import numpy as np
import pytest

from process_bigraph import Composite, gather_emitter_results

from viva_fenics import fem
from viva_fenics.core import build_core


# --------------------------------------------------------------------------- #
# fem.advection_diffusion_step
# --------------------------------------------------------------------------- #

def test_zero_velocity_matches_diffusion_step():
    """velocity=None, soret=None -> the extension is the strict generalisation
    of diffusion_step (bit-for-bit, so every existing study is unaffected)."""
    domain, V = fem.build_mesh("unit_square", 16, degree=1)
    coords = fem.node_coords(V)
    u0 = np.exp(-((coords[:, 0] - 0.5) ** 2 + (coords[:, 1] - 0.5) ** 2) / 0.02)
    src = np.zeros_like(u0)

    u_diff = fem.diffusion_step(domain, V, u0, src, 0.01, 0.1)
    u_adv = fem.advection_diffusion_step(domain, V, u0, src, 0.01, 0.1,
                                         velocity=None, soret=None)
    assert np.allclose(u_diff, u_adv, atol=1e-12, rtol=0)


def _steady_adv_diff(resolution, wx, D, supg):
    """Run backward-Euler advection-diffusion to steady state on the 1-D
    boundary-layer problem u(0)=0, u(1)=1, source 0 (uniform +x flow)."""
    domain, V = fem.build_mesh("unit_square", resolution, degree=1)
    coords = fem.node_coords(V)
    u = coords[:, 0].copy()  # linear initial guess
    src = np.zeros_like(u)
    for _ in range(400):
        u = fem.advection_diffusion_step(
            domain, V, u, src, 0.05, D, velocity=(wx, 0.0),
            bc_value={"x0": 0.0, "x1": 1.0}, supg=supg,
        )
    exact = fem.advection_diffusion_mms_bc(domain, V, wx, D, resolution)
    return fem.l2_error(domain, V, u, exact)


def test_advection_diffusion_mms_accurate_and_converges():
    """The real solve reproduces the analytic advection-diffusion boundary-layer
    solution u(x)=(exp(Pe x)-1)/(exp(Pe)-1), and the error shrinks under mesh
    refinement (the dolfinx FEM convergence witness, section 4.5 gate (a))."""
    wx, D = 1.0, 0.1  # Pe = 10; cell-Pe < 1 at these resolutions
    e_coarse = _steady_adv_diff(16, wx, D, supg=False)
    e_fine = _steady_adv_diff(32, wx, D, supg=False)
    assert e_fine < e_coarse           # convergence
    assert e_fine < 5e-3               # genuinely accurate


def test_supg_suppresses_high_peclet_oscillation():
    """At a cell Peclet number > 1 a raw Galerkin advection term produces
    node-to-node oscillations (min dips well below the min BC value of 0);
    SUPG keeps the solution monotone (near the [0,1] bounds)."""
    wx, D, res = 1.0, 0.004, 16    # cell-Pe = wx*h/D = (1/16)/0.004 ~ 15.6
    domain, V = fem.build_mesh("unit_square", res, degree=1)
    coords = fem.node_coords(V)
    src = np.zeros(coords.shape[0])

    u_no = coords[:, 0].copy()
    u_su = coords[:, 0].copy()
    for _ in range(200):
        u_no = fem.advection_diffusion_step(domain, V, u_no, src, 0.1, D,
                                            velocity=(wx, 0.0),
                                            bc_value={"x0": 0.0, "x1": 1.0}, supg=False)
        u_su = fem.advection_diffusion_step(domain, V, u_su, src, 0.1, D,
                                            velocity=(wx, 0.0),
                                            bc_value={"x0": 0.0, "x1": 1.0}, supg=True)
    assert u_no.min() < -0.05          # unstabilised overshoots below 0
    assert u_su.min() > -0.02          # SUPG stays (nearly) monotone
    assert u_su.max() < 1.05


def test_advection_translates_field_downstream():
    """A uniform +x flow advects the field's centre of mass downstream."""
    domain, V = fem.build_mesh("unit_square", 40, degree=1)
    coords = fem.node_coords(V)
    u = np.exp(-((coords[:, 0] - 0.3) ** 2 + (coords[:, 1] - 0.5) ** 2) / 0.01)
    src = np.zeros_like(u)
    xcom0 = float(np.sum(u * coords[:, 0]) / np.sum(u))
    for _ in range(20):
        u = fem.advection_diffusion_step(domain, V, u, src, 0.01, 0.01,
                                         velocity=(1.0, 0.0), supg=True)
    xcom1 = float(np.sum(u * coords[:, 0]) / np.sum(u))
    assert xcom1 > xcom0 + 0.05        # moved downstream


def test_soret_drift_conserves_and_concentrates():
    """Thermophoretic (Soret) drift under a linear cross-domain temperature
    gradient, no-flux boundaries: the field integral is conserved (the trap
    only redistributes mass) while mass concentrates on the cold (x=0) side --
    the og-03 thermal-trap behaviour, species-specific via soret."""
    domain, V = fem.build_mesh("unit_square", 32, degree=1)
    coords = fem.node_coords(V)
    u = np.ones(coords.shape[0])                     # uniform start
    src = np.zeros_like(u)
    # Linear temperature field: cold at x=0, hot at x=1 (grad(T) divergence-free).
    T = 300.0 + 40.0 * coords[:, 0]

    m0 = fem.field_integral(domain, V, u)
    xcom0 = float(np.sum(u * coords[:, 0]) / np.sum(u))
    for _ in range(80):
        u = fem.advection_diffusion_step(domain, V, u, src, 0.5, 1.0e-3,
                                         velocity=None, soret=0.1,
                                         temperature_array=T)
    m1 = fem.field_integral(domain, V, u)
    xcom1 = float(np.sum(u * coords[:, 0]) / np.sum(u))

    assert abs(m1 - m0) / m0 < 1e-9    # mass conserved to solver tolerance
    assert xcom1 < xcom0 - 0.05        # concentrated toward the cold (x=0) wall
    assert u.min() >= -1e-6            # non-negative


# --------------------------------------------------------------------------- #
# AdvectionDiffusionProcess + reactive_transport composite
# --------------------------------------------------------------------------- #

def test_process_registered():
    assert "AdvectionDiffusionProcess" in build_core().link_registry


def test_process_reduces_to_diffusion_when_still():
    """velocity_y=0, negligible reaction -> the process conserves mass under
    pure diffusion (Neumann boundary), matching DiffusionProcess."""
    core = build_core()
    from viva_fenics.composites.reactive_transport import reactive_transport
    # r=1e-9 (not 0.0, which the framework coerces to the config default) and
    # velocity_y=0.0 -> pure diffusion.
    sim = Composite({"state": reactive_transport(core=core, velocity_y=0.0, r=1e-9, D=0.05, dt=0.01)},
                    core=core)
    sim.run(0.2)
    res = gather_emitter_results(sim)[("emitter",)]
    integrals = [row["integral"] for row in res if row.get("integral")]
    assert len(integrals) > 1
    # Pure diffusion, no advection/reaction: mass conserved (Neumann boundary).
    assert abs(integrals[-1] - integrals[0]) / integrals[0] < 1e-3


def test_reactive_transport_composite_advects_and_reacts():
    """The composite couples advection (upflow) + diffusion + logistic reaction:
    the field integral grows (reaction) and the centre of mass rises (+y flow)."""
    core = build_core()
    from viva_fenics.composites.reactive_transport import reactive_transport
    res_32 = 32
    _, Vmesh = fem.build_mesh("unit_square", res_32, degree=1)
    coords = fem.node_coords(Vmesh)
    sim = Composite({"state": reactive_transport(core=core, resolution=res_32,
                                                 velocity_y=0.3, r=1.0, D=0.05, dt=0.01)},
                    core=core)
    sim.run(0.1)
    res = gather_emitter_results(sim)[("emitter",)]
    rows = [row for row in res if np.asarray(row.get("solution", [])).size]
    assert len(rows) > 1
    sol0 = np.asarray(rows[0]["solution"])
    sol1 = np.asarray(rows[-1]["solution"])
    ycom0 = float(np.sum(sol0 * coords[:, 1]) / np.sum(sol0))
    ycom1 = float(np.sum(sol1 * coords[:, 1]) / np.sum(sol1))
    assert sol1.sum() > sol0.sum()         # reaction grew the population
    assert ycom1 > ycom0 + 0.02            # upflow carried it up
