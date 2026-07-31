from process_bigraph import Process, Step, allocate_core

from pbg_fenics.processes.poisson import PoissonSolverStep


def test_poisson_step_update():
    core = allocate_core()
    step = PoissonSolverStep(config={"resolution": 16, "degree": 2}, core=core)
    out = step.update({})
    assert out["l2_error"] < 1e-10
    assert len(out["solution"]) > 0


def test_poisson_generator_registered():
    from viva_superpowers.composite_generator import _REGISTRY
    assert any(e.endswith(".poisson_baseline") for e in _REGISTRY)
