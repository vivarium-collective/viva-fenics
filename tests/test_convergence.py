import numpy as np
from process_bigraph import Composite, allocate_core, gather_emitter_results

from pbg_fenics.processes.poisson import PoissonSolverStep


def test_convergence_rate():
    core = allocate_core()
    errs = []
    for n in (8, 16, 32):
        e = PoissonSolverStep(config={"resolution": n, "degree": 1}, core=core).update({})["l2_error"]
        errs.append(e)
    rates = [np.log2(errs[i]/errs[i+1]) for i in range(len(errs)-1)]
    assert min(rates) > 1.7   # P1 -> O(h^2)


def test_mesh_convergence_generator_registered():
    from viva_superpowers.composite_generator import _REGISTRY
    assert any(e.endswith(".mesh_convergence") for e in _REGISTRY)


def test_mesh_convergence_builds():
    from pbg_fenics.core import build_core
    from pbg_fenics.composites.convergence import mesh_convergence

    core = build_core()
    doc = mesh_convergence(core, resolution=8)
    sim = Composite({"state": doc}, core=core)
    sim.run(0.0)

    res = gather_emitter_results(sim)[("emitter",)]
    l2_errors = [row["l2_error"] for row in res if row.get("l2_error")]
    assert len(l2_errors) >= 1
    assert l2_errors[0] > 0.0
