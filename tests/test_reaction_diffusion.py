import numpy as np
from process_bigraph import Composite, gather_emitter_results

from pbg_fenics.core import build_core
from pbg_fenics.composites.reaction_diffusion import reaction_diffusion
from pbg_fenics.processes.reaction import LogisticReactionProcess


def test_fisher_kpp_front_grows():
    """Headline composability proof: DiffusionProcess and
    LogisticReactionProcess are wired together only through shared bigraph
    stores (``stores.solution`` / ``stores.source``) -- neither process knows
    about the other. Running the composite should show growing total mass
    (integral), since the logistic reaction adds mass while diffusion only
    redistributes it.
    """
    core = build_core()
    doc = reaction_diffusion(core, resolution=24, D=0.05, r=2.0, dt=0.01)
    sim = Composite({"state": doc}, core=core)
    sim.run(0.2)

    res = gather_emitter_results(sim)[("emitter",)]
    integrals = [row["integral"] for row in res if row.get("integral")]
    assert len(integrals) > 1, "expected multiple post-tick emitted readings"

    first, last = integrals[0], integrals[-1]
    assert last > first  # reaction grows total mass; diffusion spreads it


def test_reaction_diffusion_generator_registered():
    from viva_superpowers.composite_generator import _REGISTRY
    assert any(e.endswith(".reaction_diffusion") for e in _REGISTRY)


def test_logistic_reaction_update_zero_field():
    core = build_core()
    p = LogisticReactionProcess(config={"r": 1.0, "K": 1.0}, core=core)
    u = np.zeros(10)
    out = p.update({"solution": u}, interval=0.01)
    assert np.allclose(out["source"], 0.0)


def test_logistic_reaction_update_grows_between_zero_and_K():
    core = build_core()
    p = LogisticReactionProcess(config={"r": 1.0, "K": 1.0}, core=core)
    u = np.full(10, 0.5)
    out = p.update({"solution": u}, interval=0.01)
    assert np.all(np.array(out["source"]) > 0)


def test_logistic_reaction_update_saturates_at_K():
    core = build_core()
    p = LogisticReactionProcess(config={"r": 1.0, "K": 1.0}, core=core)
    u = np.full(10, 1.0)
    out = p.update({"solution": u}, interval=0.01)
    assert np.allclose(out["source"], 0.0, atol=1e-10)


def test_logistic_reaction_update_no_interval_scaling():
    """The reaction outputs a pure RATE field -- no *interval and no *dt --
    since DiffusionProcess itself multiplies source by dt internally. If
    LogisticReactionProcess also scaled by interval, the reaction term would
    be applied twice (interval * dt instead of just dt).
    """
    core = build_core()
    p = LogisticReactionProcess(config={"r": 2.0, "K": 1.0}, core=core)
    u = np.full(10, 0.5)
    out_small = p.update({"solution": u}, interval=0.001)
    out_large = p.update({"solution": u}, interval=1.0)
    assert np.allclose(out_small["source"], out_large["source"])
    expected = 2.0 * 0.5 * (1 - 0.5 / 1.0)
    assert np.allclose(out_small["source"], expected)


def test_source_store_does_not_spuriously_accumulate():
    """Regression guard: LogisticReactionProcess.outputs()["source"] must be
    ``overwrite[array[float]]``. If it were a bare additive ``array[float]``,
    the shared ``stores.source`` would accumulate the instantaneous rate
    every tick instead of being replaced, and the coupled field would blow up
    well past K instead of settling near it.
    """
    core = build_core()
    doc = reaction_diffusion(core, resolution=16, D=0.05, r=2.0, dt=0.01)
    sim = Composite({"state": doc}, core=core)
    sim.run(0.3)

    res = gather_emitter_results(sim)[("emitter",)]
    for row in res:
        solution = row.get("solution")
        if solution is None:
            continue
        assert np.max(np.asarray(solution)) <= 1.5  # K=1.0 by default here
