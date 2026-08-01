import numpy as np
from process_bigraph import Process, Step, allocate_core

from viva_fenics.processes.poisson import PoissonSolverStep


def test_poisson_step_update():
    core = allocate_core()
    step = PoissonSolverStep(config={"resolution": 16, "degree": 2}, core=core)
    out = step.update({})
    assert out["l2_error"] < 1e-10
    assert len(out["solution"]) > 0


def test_poisson_generator_registered():
    from viva_superpowers.composite_generator import _REGISTRY
    assert any(e.endswith(".poisson_baseline") for e in _REGISTRY)


def test_high_order_verification_generator_registered():
    from viva_superpowers.composite_generator import _REGISTRY
    assert any(e.endswith(".high_order_verification") for e in _REGISTRY)


def _smooth_trig_rate(core, degree, resolutions):
    """Real dolfinx solves at two/three small resolutions -- fast (<1s each)
    -- fit a log-log L2-error-vs-h rate, same measurement the
    poisson-validation study's sims/run.py performs at production scale."""
    errors = []
    for n in resolutions:
        step = PoissonSolverStep(
            config={"resolution": n, "degree": degree, "problem": "smooth_trig"},
            core=core,
        )
        errors.append(step.update({})["l2_error"])
    h = [1.0 / n for n in resolutions]
    slope, _intercept = np.polyfit(np.log(h), np.log(errors), 1)
    return float(slope), errors


def test_poisson_step_smooth_trig_problem_runs():
    core = allocate_core()
    step = PoissonSolverStep(
        config={"resolution": 16, "degree": 2, "problem": "smooth_trig"}, core=core,
    )
    out = step.update({})
    # Not exactly representable (non-polynomial exact solution) -- error
    # must be genuinely non-trivial, not round-off, and not huge.
    assert 1e-8 < out["l2_error"] < 1e-1
    assert len(out["solution"]) > 0


def test_poisson_smooth_trig_p1_achieves_optimal_rate():
    core = allocate_core()
    rate, errors = _smooth_trig_rate(core, degree=1, resolutions=(16, 32, 48))
    assert errors[0] > errors[-1] > 0.0  # error must genuinely decrease
    assert abs(rate - 2.0) <= 0.25  # theoretical optimal L2 rate for P1


def test_poisson_smooth_trig_p2_achieves_optimal_rate():
    core = allocate_core()
    rate, errors = _smooth_trig_rate(core, degree=2, resolutions=(16, 32, 48))
    assert errors[0] > errors[-1] > 0.0
    assert abs(rate - 3.0) <= 0.25  # theoretical optimal L2 rate for P2


def test_poisson_smooth_trig_p3_achieves_optimal_rate():
    core = allocate_core()
    rate, errors = _smooth_trig_rate(core, degree=3, resolutions=(16, 32, 48))
    assert errors[0] > errors[-1] > 0.0
    assert abs(rate - 4.0) <= 0.25  # theoretical optimal L2 rate for P3


def test_high_order_verification_builds():
    from process_bigraph import Composite, gather_emitter_results

    from viva_fenics.core import build_core
    from viva_fenics.composites.poisson import high_order_verification

    core = build_core()
    doc = high_order_verification(core, degree=2, resolution=16)
    sim = Composite({"state": doc}, core=core)
    sim.run(0.0)

    rows = gather_emitter_results(sim)[("emitter",)]
    l2_errors = [row["l2_error"] for row in rows if row.get("l2_error")]
    assert len(l2_errors) >= 1
    assert l2_errors[0] > 0.0
