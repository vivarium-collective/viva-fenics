import numpy as np
from process_bigraph import Composite, gather_emitter_results

from viva_fenics.core import build_core
from viva_fenics.composites.reaction_diffusion import reaction_diffusion
from viva_fenics.processes.reaction import LogisticReactionProcess


def _integral_growth(r, resolution=24, D=0.05, dt=0.01, run_time=0.2):
    """Build+run a reaction_diffusion composite at the given r and return
    (last_integral - first_integral) over post-tick emitted readings."""
    core = build_core()
    doc = reaction_diffusion(core, resolution=resolution, D=D, r=r, dt=dt)
    sim = Composite({"state": doc}, core=core)
    sim.run(run_time)

    res = gather_emitter_results(sim)[("emitter",)]
    integrals = [row["integral"] for row in res if row.get("integral")]
    assert len(integrals) > 1, "expected multiple post-tick emitted readings"
    return integrals[-1] - integrals[0]


def test_fisher_kpp_front_grows():
    """Headline composability proof: DiffusionProcess and
    LogisticReactionProcess are wired together only through shared bigraph
    stores (``stores.solution`` / ``stores.source``) -- neither process knows
    about the other.

    A bare "last > first" check on a single r=2.0 run is NOT discriminating:
    it also passes with the reaction effectively disconnected (verified: at
    r~=0 the integral still moves by 5-20 milli-units purely from
    solver/quadrature noise over these run lengths). Instead, compare mass
    growth WITH the reaction (r=2.0) against a near-zero-reaction control
    (r=1e-9, same params/wiring) -- only a genuinely-coupled reaction should
    produce growth dramatically (many orders of magnitude) larger than the
    r~=0 control.

    NOTE on r=1e-9 vs r=0.0: the installed bigraph-schema's `is_empty` for a
    plain `float` leaf treats an explicit ``0.0`` as "empty" (same as
    ``None``, see bigraph_schema/methods/is_empty.py's
    ``is_empty(Float, value) = value is None or value == 0.0``), so
    Composite's `core.fill(config_schema, {"r": 0.0})` silently discards the
    provided ``0.0`` and substitutes the config_schema default (``r=1.0``)
    instead -- confirmed directly: ``core.fill({"r": {"_type": "float",
    "_default": 1.0}}, {"r": 0.0})`` returns ``{"r": 1.0}``. This is a
    framework-level quirk (out of scope to change here; it does not involve
    reaction.py or the generator), and it is exactly why an earlier version
    of this test using ``r=0.0`` as the "off" control was not actually
    testing r=0 -- it was silently testing r=1.0 vs r=2.0, a 2x difference,
    not a real disconnect. ``r=1e-9`` is not exactly ``0.0`` so it passes
    through `fill` unmodified and gives a genuine, physically negligible
    reaction contribution.
    """
    growth_r2 = _integral_growth(r=2.0, run_time=0.5)
    growth_r_off = _integral_growth(r=1e-9, run_time=0.5)

    assert growth_r2 > 5 * abs(growth_r_off)
    assert growth_r2 > 0.05


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

    Run long enough (t=0.8) and with a tight enough bound (1.2, vs K=1.0)
    that the additive-accumulation bug provably fails this: at these params
    (resolution=16, r=2.0, dt=0.01) the bare-additive version reaches a
    max field >> 1.2 by t=0.8, while the correct overwrite version stays
    bounded near K throughout (logistic saturation caps growth once the
    field approaches K, regardless of run length).
    """
    core = build_core()
    doc = reaction_diffusion(core, resolution=16, D=0.05, r=2.0, dt=0.01)
    sim = Composite({"state": doc}, core=core)
    sim.run(0.8)

    res = gather_emitter_results(sim)[("emitter",)]
    max_field = 0.0
    for row in res:
        solution = row.get("solution")
        if solution is None:
            continue
        max_field = max(max_field, float(np.max(np.asarray(solution))))
    assert max_field < 1.2  # K=1.0 by default here
