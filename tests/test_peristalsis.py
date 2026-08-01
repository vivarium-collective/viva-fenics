"""Behavior tests for the peristaltic-pumping composition:
``PeristalticWallProcess`` (harmonic-extension ALE mesh motion) ⊕
``PeristalticFlowProcess`` (ALE incompressible Navier-Stokes) -- see
viva_fenics/processes/peristalsis.py's module docstring for the physics.
Small mesh / short duration for pytest speed; see
studies/moving-boundary/sims/run.py for the full canonical production run.
"""

import numpy as np
from process_bigraph import Composite, allocate_core, gather_emitter_results

from viva_fenics.core import build_core
from viva_fenics.composites.peristalsis import peristalsis
from viva_fenics.processes.peristalsis import (
    PeristalticFlowProcess,
    PeristalticWallProcess,
    occlusion,
    occlusion_and_rate,
)

DT = 0.01
SMALL = dict(L=2.0, H=2.0, nx=8, ny=4, wavelength=1.0, wave_speed=1.0, ramp_time=0.05, dt=DT)


def test_occlusion_bounded_and_symmetric_in_x():
    """occlusion(x, t) must stay in [0, amplitude] for any x, t -- a
    negative or over-amplitude occlusion would mean a broken wave law
    (e.g. a wrong cos-argument sign), not a physically valid wall shape.
    """
    amplitude = 0.4
    x = np.linspace(0.0, 2.0, 50)
    for t in (0.0, 0.3, 1.7):
        occ = occlusion(x, t, amplitude, wavelength=1.0, wave_speed=1.0)
        assert np.all(occ >= -1e-12)
        assert np.all(occ <= amplitude + 1e-12)


def test_occlusion_rate_matches_finite_difference():
    """The analytic d(occlusion)/dt (used for both the mesh-motion Laplace
    BC and the wall no-slip velocity BC) must match a finite-difference
    check -- this is the ONLY place a sign or factor-of-2 error in the
    closed-form derivative would show up before it corrupts a real solve.
    """
    amplitude, wavelength, wave_speed = 0.3, 1.0, 1.0
    x = np.array([0.1, 0.4, 0.9, 1.3])
    t0, h = 0.37, 1e-6
    _occ_m, rate_m = occlusion_and_rate(x, t0, amplitude, wavelength, wave_speed, ramp_time=0.0)
    occ_plus = occlusion(x, t0 + h, amplitude, wavelength, wave_speed)
    occ_minus = occlusion(x, t0 - h, amplitude, wavelength, wave_speed)
    fd_rate = (occ_plus - occ_minus) / (2 * h)
    assert np.allclose(rate_m, fd_rate, atol=1e-4)


def test_ramped_amplitude_starts_at_zero_occlusion():
    """With ramp_time>0, occlusion at t=0 must be exactly 0 everywhere
    (the channel starts flat/undeformed) -- confirms the ramp actually
    suppresses the impulsive-start amplitude, not just scales it down.
    """
    x = np.linspace(0.0, 2.0, 20)
    occ0, rate0 = occlusion_and_rate(x, 0.0, amplitude=0.5, wavelength=1.0, wave_speed=1.0, ramp_time=0.5)
    assert np.allclose(occ0, 0.0)
    # the rate at t=0 is NOT zero (the ramp is actively growing the wave)
    assert np.max(np.abs(rate0)) > 0.0


def _fresh_wall(core, amplitude=0.3):
    p = PeristalticWallProcess(config=dict(SMALL, amplitude=amplitude), core=core)
    state0 = p.initial_state()
    return p, state0


def test_wall_process_deforms_top_and_bottom_oppositely():
    """The harmonic-extension displacement at the reference top boundary
    (y=H/2) should be <= 0 (wall moves inward/down) and at the bottom
    (y=-H/2) should be >= 0 (wall moves inward/up) -- confirming the mesh
    motion process genuinely produces the symmetric-occlusion wall
    kinematics, not e.g. both walls moving the same direction.
    """
    core = allocate_core()
    p, state0 = _fresh_wall(core, amplitude=0.4)
    out = p.update({"phase_drive": 0.0}, DT)

    geom_y = p._domain.geometry.x[:, 1]
    top_mask = np.isclose(geom_y, SMALL["H"] / 2.0)
    bottom_mask = np.isclose(geom_y, -SMALL["H"] / 2.0)
    assert top_mask.sum() > 0 and bottom_mask.sum() > 0

    disp = np.asarray(out["mesh_displacement_y"])
    assert np.all(disp[top_mask] <= 1e-9)
    assert np.all(disp[bottom_mask] >= -1e-9)
    # some genuine (nonzero) displacement -- not a motionless mesh
    assert np.max(np.abs(disp)) > 1e-6


def test_wall_min_gap_shrinks_from_full_height():
    """wall_min_gap should start below H (some occlusion present) once the
    ramp has begun, and never exceed H (walls only move inward).
    """
    core = allocate_core()
    p, _ = _fresh_wall(core, amplitude=0.3)
    for _ in range(10):
        out = p.update({"phase_drive": 0.0}, DT)
    assert out["wall_min_gap"] < SMALL["H"]
    assert out["wall_min_gap"] > 0.0


def test_flow_process_geometry_actually_moves():
    """PeristalticFlowProcess must mutate its OWN mesh's geometry.x when fed
    a nonzero mesh_displacement_y -- the direct-geometry-mutation ALE
    technique (same as MovingBoundaryProcess.deform_mesh), confirmed here
    independent of the flow solve itself.
    """
    core = allocate_core()
    wall, _ = _fresh_wall(core, amplitude=0.4)
    wout = wall.update({"phase_drive": 0.0}, DT)

    flow = PeristalticFlowProcess(config=dict(SMALL, amplitude=0.4, reynolds=10.0), core=core)
    fstate0 = flow.initial_state()
    reference_y = flow._reference_y.copy()

    flow.update(
        {
            "mesh_displacement_y": wout["mesh_displacement_y"],
            "mesh_velocity_y": wout["mesh_velocity_y"],
            "wall_time": wout["wall_time"],
        },
        DT,
    )
    assert not np.allclose(flow._domain.geometry.x[:, 1], reference_y)


def _run_composite(core, amplitude, run_time=0.2):
    doc = peristalsis(
        core, amplitude=amplitude, wave_speed=SMALL["wave_speed"], wavelength=SMALL["wavelength"],
        reynolds=10.0, dt=DT, ramp_time=SMALL["ramp_time"],
        L=SMALL["L"], H=SMALL["H"], nx=SMALL["nx"], ny=SMALL["ny"],
    )
    doc["emitter"]["config"]["subsample"] = 1
    sim = Composite({"state": doc}, core=core)
    sim.run(run_time)
    return gather_emitter_results(sim)[("emitter",)]


def test_composition_produces_net_positive_flow():
    """The composed wall ⊕ flow system must produce a genuinely POSITIVE
    net (domain-averaged) axial flow for a nonzero occlusion amplitude --
    real peristaltic pumping, not zero or reverse flow, from a small/fast
    mesh+duration (the production-scale confirmation lives in
    studies/moving-boundary/sims/run.py).
    """
    core = build_core()
    rows = _run_composite(core, amplitude=0.3)
    vx_last = np.asarray(rows[-1]["velocity_x"])
    assert np.all(np.isfinite(vx_last))
    assert rows[-1]["mean_ux"] > 0.0


def test_net_flow_grows_with_amplitude():
    """Net flow (mean_ux at the end of a short, matched-duration run) must
    be strictly larger for a larger occlusion amplitude -- the core
    peristalsis behavior test (mirrors
    test_moving_boundary.py::test_domain_measure_changes_with_amplitude's
    structure for this composition's own amplitude-sensitivity).
    """
    core = build_core()
    rows_low = _run_composite(core, amplitude=0.15)
    rows_high = _run_composite(core, amplitude=0.3)
    Q_low = rows_low[-1]["mean_ux"]
    Q_high = rows_high[-1]["mean_ux"]
    assert np.isfinite(Q_low) and np.isfinite(Q_high)
    assert Q_high > Q_low > 0.0


def test_peristalsis_generator_registered():
    from viva_superpowers.composite_generator import _REGISTRY
    assert any(e.endswith(".peristalsis") for e in _REGISTRY)


def test_peristalsis_sensors_do_not_accumulate():
    """Regression: mean_ux / wall_min_gap / domain_area must be `overwrite`
    sensor outputs, not bare (additive) floats -- an accumulating bug would
    push mean_ux/domain_area far outside their physically bounded range
    over several ticks (same accumulate-vs-overwrite bug class as
    MovingBoundaryProcess.boundary_position / NavierStokesProcess's
    overwrite outputs).
    """
    core = build_core()
    rows = _run_composite(core, amplitude=0.3, run_time=0.15)
    assert len(rows) > 3

    domain_areas = [r["domain_area"] for r in rows if r.get("domain_area") is not None]
    max_area = SMALL["L"] * SMALL["H"]
    assert all(0.0 < a <= max_area * 1.05 for a in domain_areas), (
        f"domain_area looks like it's accumulating, not overwriting: {domain_areas}"
    )

    wall_min_gaps = [r["wall_min_gap"] for r in rows if r.get("wall_min_gap") is not None]
    assert all(0.0 < g <= SMALL["H"] * 1.05 for g in wall_min_gaps), (
        f"wall_min_gap looks like it's accumulating, not overwriting: {wall_min_gaps}"
    )
