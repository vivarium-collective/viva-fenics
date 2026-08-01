"""Behavior tests for MovingBoundaryProcess (real dolfinx diffusion re-solved
on a prescribed-ALE deforming mesh -- see
viva_fenics/processes/moving_boundary.py's module docstring for the scheme).
Small resolution/short duration for pytest speed; see
studies/moving-boundary/sims/run.py for the full canonical run.
"""

import numpy as np
from process_bigraph import allocate_core

from viva_fenics.processes.moving_boundary import (
    MovingBoundaryProcess,
    boundary_law,
    domain_area,
)

RESOLUTION = 12
DT = 0.01


def _fresh_process(mode="oscillate", amplitude=0.3, omega=3.14159, resolution=RESOLUTION):
    core = allocate_core()
    p = MovingBoundaryProcess(
        config={
            "resolution": resolution,
            "degree": 1,
            "amplitude": amplitude,
            "omega": omega,
            "D": 0.1,
            "dt": DT,
            "mode": mode,
        },
        core=core,
    )
    solution0 = p.initial_state()["solution"]
    return p, solution0


def test_boundary_position_follows_oscillate_law():
    """For mode='oscillate', boundary_position at time t should sit close
    to the analytic law 1 + amplitude*sin(omega*t) at a few checkpoints --
    confirming the mesh really is being deformed according to the
    prescribed law, not drifting or staying fixed.
    """
    amplitude, omega = 0.3, 3.14159
    p, solution0 = _fresh_process(mode="oscillate", amplitude=amplitude, omega=omega)

    t = 0.0
    for interval in (0.1, 0.15, 0.2):
        out = p.update({"solution": solution0, "boundary_drive": 0.0}, interval=interval)
        solution0 = solution0 + np.array(out["solution"])
        t += interval
        expected = boundary_law("oscillate", t, amplitude, omega)
        assert abs(out["boundary_position"] - expected) < 1e-6, (
            f"boundary_position {out['boundary_position']} != analytic {expected} at t={t}"
        )


def test_boundary_position_monotonic_for_grow_mode():
    """mode='grow' (y_top = 1 + amplitude*t) should produce a strictly
    monotonically increasing boundary_position across successive ticks.
    """
    p, solution0 = _fresh_process(mode="grow", amplitude=0.5)

    positions = []
    for _ in range(4):
        out = p.update({"solution": solution0, "boundary_drive": 0.0}, interval=0.1)
        solution0 = solution0 + np.array(out["solution"])
        positions.append(out["boundary_position"])

    assert all(b > a for a, b in zip(positions, positions[1:])), (
        f"boundary_position should be strictly increasing under mode='grow': {positions}"
    )


def test_domain_measure_tracks_boundary_position():
    """The domain is exactly [0,1] x [0, y_top(t)] (only the height moves;
    width stays 1), so the FEM-assembled domain_measure (integral(1) dx
    over the deformed mesh) should equal boundary_position to near machine
    precision -- a strong, non-tautological check that the mesh geometry
    itself was really mutated, not just a scalar tracked in isolation.
    """
    p, solution0 = _fresh_process(mode="oscillate", amplitude=0.4, omega=3.14159)
    out = p.update({"solution": solution0, "boundary_drive": 0.0}, interval=0.2)

    assert abs(out["domain_measure"] - out["boundary_position"]) < 1e-8, (
        f"domain_measure {out['domain_measure']} should equal boundary_position "
        f"{out['boundary_position']} for a unit-width domain"
    )


def test_domain_measure_changes_with_amplitude():
    """A larger oscillation amplitude should produce a larger swing in
    domain_measure away from the base area (1.0) over the same run.
    """
    def _max_deviation(amplitude):
        p, solution0 = _fresh_process(mode="oscillate", amplitude=amplitude, omega=3.14159)
        deviations = []
        for _ in range(20):
            out = p.update({"solution": solution0, "boundary_drive": 0.0}, interval=0.02)
            solution0 = solution0 + np.array(out["solution"])
            deviations.append(abs(out["domain_measure"] - 1.0))
        return max(deviations)

    small = _max_deviation(0.1)
    large = _max_deviation(0.5)
    assert large > small, (
        f"larger amplitude should produce a larger domain_measure swing: "
        f"amplitude=0.1 -> {small}, amplitude=0.5 -> {large}"
    )


def test_boundary_drive_offsets_the_prescribed_law():
    """A sibling-writable boundary_drive input should additively offset the
    prescribed boundary law -- confirming the port is a real, functioning
    hook, not vestigial.
    """
    amplitude, omega = 0.3, 3.14159
    p, solution0 = _fresh_process(mode="oscillate", amplitude=amplitude, omega=omega)

    drive = 0.2
    out = p.update({"solution": solution0, "boundary_drive": drive}, interval=0.1)
    expected = boundary_law("oscillate", 0.1, amplitude, omega) + drive
    assert abs(out["boundary_position"] - expected) < 1e-6


def test_field_stays_finite_and_bounded():
    """The diffusion field solved on the deforming mesh should stay finite
    and bounded (no blow-up) across a run spanning a full oscillation
    period.
    """
    p, solution0 = _fresh_process(mode="oscillate", amplitude=0.3, omega=3.14159)
    period = 2.0 * np.pi / 3.14159

    field = np.array(solution0)
    for _ in range(10):
        out = p.update({"solution": field, "boundary_drive": 0.0}, interval=period / 10.0)
        field = field + np.array(out["solution"])
        assert np.all(np.isfinite(field)), "field must stay finite through the deforming-mesh solve"
        assert np.max(np.abs(field)) < 10.0, "field should stay bounded, not blow up"


def test_domain_area_helper_matches_analytic_rectangle():
    """Direct unit check of the domain_area() FEM assembly against the
    analytic rectangle area for a few deformed configurations, independent
    of the Process wrapper.
    """
    from viva_fenics.processes.moving_boundary import _build_mesh, deform_mesh

    domain, _V, y_ref = _build_mesh(RESOLUTION)
    for y_top in (0.7, 1.0, 1.4):
        deform_mesh(domain, y_ref, y_top)
        area = domain_area(domain)
        assert abs(area - y_top) < 1e-10


def test_moving_boundary_generator_registered():
    from viva_superpowers.composite_generator import _REGISTRY
    assert any(e.endswith(".moving_boundary") for e in _REGISTRY)


def test_moving_boundary_composite_sensors_do_not_accumulate():
    """Regression: boundary_position/domain_measure must be `overwrite`
    sensor outputs, not bare (additive) floats -- a bare output composes
    additively in the store's apply, so a single writer ticking N times
    would accumulate sum_i value_i instead of reporting the true current
    absolute reading (same accumulate-vs-overwrite bug class as
    DiffusionProcess.integral / NavierStokesProcess's overwrite outputs).
    """
    from process_bigraph import Composite, gather_emitter_results

    from viva_fenics.composites.moving_boundary import moving_boundary

    core = allocate_core()
    core.register_link("MovingBoundaryProcess", MovingBoundaryProcess)

    document = moving_boundary(core=core, resolution=8, amplitude=0.3, dt=0.02)
    sim = Composite({"state": document}, core=core)
    sim.run(0.1)  # several process ticks (interval=dt=0.02 -> 5 ticks)

    rows = gather_emitter_results(sim)[("emitter",)]
    positions = [row["boundary_position"] for row in rows if row.get("boundary_position")]
    assert len(positions) > 1, "expected multiple post-tick emitted readings"

    # An accumulating bug would push later readings far outside the
    # physically bounded range of ``1 + amplitude*sin(...)`` (~[0.7, 1.3]
    # here); an overwritten sensor should stay within that range.
    assert all(0.5 < p < 1.5 for p in positions), (
        f"boundary_position looks like it's accumulating, not overwriting: {positions}"
    )
