from process_bigraph import allocate_core
from pbg_fenics.processes.diffusion import DiffusionProcess
import numpy as np


def test_diffusion_mass_and_smoothing():
    core = allocate_core()
    p = DiffusionProcess(config={"resolution": 32, "D": 0.1, "dt": 0.01}, core=core)
    s0 = np.array(p.initial_state()["solution"])
    delta = p.update({"source": np.zeros_like(s0), "solution": s0}, interval=0.05)
    s1 = s0 + np.array(delta["solution"])
    assert s1.max() < s0.max()            # peak diffuses down
    assert abs(s1.sum() - s0.sum()) / s0.sum() < 0.05   # ~mass conserved (no-flux/decay)


def test_diffusion_generator_registered():
    from viva_superpowers.composite_generator import _REGISTRY
    assert any(e.endswith(".transient_diffusion") for e in _REGISTRY)


def test_diffusion_composite_integral_does_not_accumulate():
    """Regression: `integral` must be an `overwrite[float]` sensor reading,
    not a bare (additive) float.

    A bare float output composes additively in the store's apply, so a
    single writer ticking N times would accumulate
    sum_i integral_i (~= N x true mass) instead of reporting the true
    absolute per-tick integral. This exercises the real Composite apply
    path (unlike test_diffusion_mass_and_smoothing, which calls
    DiffusionProcess.update() directly and never touches store apply).
    """
    from process_bigraph import Composite, allocate_core, gather_emitter_results

    from pbg_fenics.composites.diffusion import transient_diffusion

    core = allocate_core()
    # pbg-fenics is an editable (hatchling) install, so importlib.metadata's
    # packages_distributions() has no entry for it and allocate_core()'s
    # discovery walk skips it entirely -- a pre-existing, repo-wide gap
    # (PoissonSolverStep has it too), unrelated to this fix and out of
    # scope to change here. Register the local process explicitly so
    # `local:DiffusionProcess` resolves for this test's Composite.
    core.register_link("DiffusionProcess", DiffusionProcess)

    document = transient_diffusion(core=core, resolution=8, D=0.1, dt=0.01)
    sim = Composite({"state": document}, core=core)
    sim.run(3.0)  # several process ticks (default process interval = 1.0)

    rows = gather_emitter_results(sim)[("emitter",)]
    # The emitter fires once at t=0 (before the process has ticked at all,
    # so "integral" is still its unset 0.0 default) and once per process
    # tick thereafter. Drop the pre-run row -- a real 0.0 integral never
    # occurs post-tick for a non-degenerate gaussian bump, so filtering on
    # truthiness is an unambiguous, non-magic-index way to isolate them.
    integrals = [row["integral"] for row in rows if row.get("integral")]
    assert len(integrals) > 1, "expected multiple post-tick emitted readings"

    first, last = integrals[0], integrals[-1]
    # No-flux diffusion with zero source conserves mass: the true integral
    # should stay ~flat, not grow tick-over-tick.
    assert abs(last - first) / first < 0.1
    # The additive-accumulation bug would give last ~= N x first (N > 2
    # here); this bounds it well below that failure mode.
    assert last < 2 * first
