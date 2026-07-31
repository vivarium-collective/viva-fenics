"""Behavior tests for NavierStokesProcess (real dolfinx IPCS lid-driven
cavity flow -- see viva_fenics/processes/flow.py's module docstring for the
scheme). Small resolution/short duration for pytest speed; see
studies/navier-stokes/sims/run.py for the full-resolution canonical run.
"""

import numpy as np
from process_bigraph import allocate_core

from viva_fenics.processes.flow import (
    NavierStokesProcess,
    divergence_stats,
    velocity_coords,
    zero_fields,
)

RESOLUTION = 12
DT = 0.01
LID_VELOCITY = 1.0


def _fresh_process(reynolds=100.0, resolution=RESOLUTION, lid_velocity=LID_VELOCITY):
    core = allocate_core()
    p = NavierStokesProcess(
        config={
            "resolution": resolution,
            "reynolds": reynolds,
            "dt": DT,
            "lid_velocity": lid_velocity,
        },
        core=core,
    )
    body_force = p.initial_state()["body_force"]
    return p, body_force


def test_steady_velocity_is_nontrivial_and_bounded():
    """A lid-driven cavity solve should produce a non-trivial flow field
    (something actually happened) whose peak speed sits near the lid
    velocity (the moving-lid Dirichlet BC pins the maximum speed close to
    lid_velocity for a standard cavity Reynolds number) rather than blowing
    up or staying at zero.
    """
    p, body_force = _fresh_process()
    out = p.update({"body_force": body_force}, interval=0.3)  # 30 substeps

    vx = np.asarray(out["velocity_x"])
    vy = np.asarray(out["velocity_y"])
    speed = np.sqrt(vx**2 + vy**2)

    assert speed.max() > 0.1 * LID_VELOCITY, "flow should be non-trivial, not near-zero"
    # The lid BC directly prescribes u=(lid_velocity, 0) there, so the
    # global max speed should sit close to lid_velocity, not blow up.
    assert speed.max() < 1.5 * LID_VELOCITY, "max speed should stay near the lid velocity, not blow up"
    assert np.all(np.isfinite(vx)) and np.all(np.isfinite(vy))
    assert out["speed_integral"] > 0.0


def test_approximately_divergence_free():
    """IPCS is only *approximately* incompressible (operator-splitting error,
    plus the lid-driven cavity's two discontinuous top-corner BCs) -- but
    the mean absolute divergence over the domain should be small relative
    to the O(1) velocity scale, and even the max (excluding the singular
    lid corners) should stay bounded, not blow up.
    """
    p, body_force = _fresh_process()
    p.update({"body_force": body_force}, interval=0.3)  # reach quasi-steady

    mean_div, max_div = divergence_stats(p._domain, p._V, p._u_n.x.array)
    assert mean_div < 0.15, f"mean|div(u)| unexpectedly large: {mean_div}"

    coords = velocity_coords(RESOLUTION)
    dist_to_top_corners = np.minimum(
        np.hypot(coords[:, 0] - 0.0, coords[:, 1] - 1.0),
        np.hypot(coords[:, 0] - 1.0, coords[:, 1] - 1.0),
    )
    # DG0 cell coords differ from V's node coords, so re-derive max|div| over
    # cells away from the lid corners directly rather than reusing `coords`.
    import ufl
    from dolfinx import fem
    DG0 = fem.functionspace(p._domain, ("DG", 0))
    u_fn = fem.Function(p._V)
    u_fn.x.array[:] = p._u_n.x.array
    div_expr = fem.Expression(ufl.div(u_fn), DG0.element.interpolation_points)
    div_field = fem.Function(DG0)
    div_field.interpolate(div_expr)
    cell_coords = DG0.tabulate_dof_coordinates()[:, :2]
    dist = np.minimum(
        np.hypot(cell_coords[:, 0] - 0.0, cell_coords[:, 1] - 1.0),
        np.hypot(cell_coords[:, 0] - 1.0, cell_coords[:, 1] - 1.0),
    )
    away_from_corners = np.abs(div_field.x.array[dist > 0.15])
    assert away_from_corners.max() < 3.0, "|div(u)| away from the lid corners should stay bounded"


def test_reaches_quasi_steady_state():
    """Run to t=0.3 (quasi-steady, per test_steady_velocity_is_nontrivial),
    then advance a further short interval and confirm the flow has stopped
    changing much -- i.e. the scheme actually converges to a steady cavity
    flow rather than oscillating or drifting indefinitely.
    """
    p, body_force = _fresh_process()
    out1 = p.update({"body_force": body_force}, interval=0.3)
    out2 = p.update({"body_force": body_force}, interval=0.05)

    speed1 = out1["speed_integral"]
    speed2 = out2["speed_integral"]
    assert speed1 > 0.0
    rel_change = abs(speed2 - speed1) / speed1
    assert rel_change < 0.05, f"speed_integral still drifting: {speed1} -> {speed2}"


def test_higher_lid_velocity_increases_speed_integral():
    """At fixed Reynolds number (so nu scales with lid_velocity, keeping the
    non-dimensional flow structure ~fixed), a faster-moving lid should
    produce a larger velocity L2 norm over the domain -- a basic physical
    sanity/trend check that this is a real, parameter-sensitive NS solve.
    """
    p_slow, bf_slow = _fresh_process(lid_velocity=1.0)
    out_slow = p_slow.update({"body_force": bf_slow}, interval=0.3)

    p_fast, bf_fast = _fresh_process(lid_velocity=2.0)
    out_fast = p_fast.update({"body_force": bf_fast}, interval=0.3)

    assert out_fast["speed_integral"] > out_slow["speed_integral"]


def test_navier_stokes_generator_registered():
    from viva_superpowers.composite_generator import _REGISTRY
    assert any(e.endswith(".navier_stokes") for e in _REGISTRY)


def test_navier_stokes_composite_outputs_do_not_accumulate():
    """Regression: velocity_x/velocity_y/pressure/speed_integral must be
    `overwrite` sensor outputs, not bare (additive) arrays/floats -- a bare
    output composes additively in the store's apply, so a single writer
    ticking N times would accumulate `sum_i field_i` instead of reporting
    the true current absolute field (the same accumulate-vs-overwrite bug
    class covered in tests/test_diffusion.py).
    """
    from process_bigraph import Composite, gather_emitter_results

    from viva_fenics.composites.flow import navier_stokes

    core = allocate_core()
    core.register_link("NavierStokesProcess", NavierStokesProcess)

    document = navier_stokes(core=core, resolution=8, reynolds=100.0, dt=0.02)
    sim = Composite({"state": document}, core=core)
    sim.run(0.1)  # several process ticks (interval=dt=0.02 -> 5 ticks)

    rows = gather_emitter_results(sim)[("emitter",)]
    speeds = [row["speed_integral"] for row in rows if row.get("speed_integral")]
    assert len(speeds) > 1, "expected multiple post-tick emitted readings"

    # An accumulating bug would make later readings grow roughly linearly
    # tick-over-tick (sum of N similar-magnitude readings); a correctly
    # overwritten sensor should instead converge/stay the same order of
    # magnitude as the first reading, not multiply by N.
    first, last = speeds[0], speeds[-1]
    assert last < 3 * first, "speed_integral looks like it's accumulating, not overwriting"


def test_zero_fields_matches_velocity_coords_length():
    vx0, vy0, p0 = zero_fields(RESOLUTION)
    coords = velocity_coords(RESOLUTION)
    assert len(vx0) == len(vy0) == coords.shape[0]
    assert np.all(vx0 == 0.0) and np.all(vy0 == 0.0) and np.all(p0 == 0.0)
