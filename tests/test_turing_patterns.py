"""Fast tests for the Gray-Scott Turing-pattern showcase: 2 x DiffusionProcess
(species U, V) ⊕ GrayScottReactionProcess, coupled purely through shared
bigraph stores (see ``viva_fenics.composites.turing_patterns``).

These are all small-mesh / few-step checks -- forming a FULL, visually clear
Turing pattern takes thousands of backward-Euler steps at production
resolution (see ``studies/reaction-diffusion/sims/run.py``), which does not
belong in the unit test suite. What's tested here instead:

1. ``GrayScottReactionProcess.update`` math is correct in isolation (rate
   signs, the U=1/V=0 background-is-inert case, no double-application of
   ``interval``).
2. The composed 3-process system has a genuine PATTERN-FORMING TENDENCY: a
   short run from the seeded initial condition grows V's spatial
   heterogeneity (variance), which can only happen if all three processes
   are actually exchanging information through ``stores.u``/``stores.v``/
   ``stores.source_u``/``stores.source_v`` -- not a claim about the mature,
   many-thousand-step pattern itself.
3. No blow-up: fields stay within a loose, physically-sane band over a
   short-to-medium run.
"""
from __future__ import annotations

import numpy as np
from process_bigraph import Composite

from viva_fenics.core import build_core
from viva_fenics.composites.turing_patterns import turing_patterns, _gray_scott_seed
from viva_fenics.processes.reaction import GrayScottReactionProcess
from viva_fenics import fem


def test_turing_patterns_generator_registered():
    from viva_superpowers.composite_generator import _REGISTRY
    assert any(e.endswith(".turing_patterns") for e in _REGISTRY)


def test_gray_scott_process_registered_in_core():
    core = build_core()
    assert core.link_registry.get("GrayScottReactionProcess") is not None


# ---------------------------------------------------------------------------
# GrayScottReactionProcess.update math
# ---------------------------------------------------------------------------

def test_gray_scott_reaction_background_is_inert():
    """At U=1, V=0 (the Gray-Scott background, no autocatalytic UV^2 term
    active) both source rates must be ~0: source_u = -0 + F*(1-1) = 0,
    source_v = 0 - (F+k)*0 = 0. If either were nonzero the uniform
    background itself would drift, which would make the "everywhere except a
    small seeded patch" initial condition meaningless.
    """
    core = build_core()
    p = GrayScottReactionProcess(config={"F": 0.037, "k": 0.06}, core=core)
    u = np.ones(12)
    v = np.zeros(12)
    out = p.update({"u": u, "v": v}, interval=1.0)
    assert np.allclose(out["source_u"], 0.0, atol=1e-12)
    assert np.allclose(out["source_v"], 0.0, atol=1e-12)


def test_gray_scott_reaction_signs_in_seeded_region():
    """In the classic seeded patch (U=0.5, V=0.25, spots regime F=0.035,
    k=0.065): U is consumed by the autocatalytic reaction (source_u < 0) and
    V is produced faster than it decays (source_v > 0) -- the combination
    that lets a local perturbation grow into a pattern instead of dying out.
    """
    core = build_core()
    p = GrayScottReactionProcess(config={"F": 0.035, "k": 0.065}, core=core)
    u = np.full(8, 0.5)
    v = np.full(8, 0.25)
    out = p.update({"u": u, "v": v}, interval=1.0)

    uvv = 0.5 * 0.25**2
    expected_source_u = -uvv + 0.035 * (1 - 0.5)
    expected_source_v = uvv - (0.035 + 0.065) * 0.25

    assert np.allclose(out["source_u"], expected_source_u)
    assert np.allclose(out["source_v"], expected_source_v)
    assert expected_source_u < 0
    assert expected_source_v > 0


def test_gray_scott_reaction_no_interval_scaling():
    """The reaction outputs pure RATE fields -- no *interval and no *dt --
    since DiffusionProcess itself multiplies each source by dt internally
    (same convention/rationale as LogisticReactionProcess; see that
    process's test of the same name).
    """
    core = build_core()
    p = GrayScottReactionProcess(config={"F": 0.037, "k": 0.06}, core=core)
    u = np.full(6, 0.5)
    v = np.full(6, 0.25)
    out_small = p.update({"u": u, "v": v}, interval=0.001)
    out_large = p.update({"u": u, "v": v}, interval=5.0)
    assert np.allclose(out_small["source_u"], out_large["source_u"])
    assert np.allclose(out_small["source_v"], out_large["source_v"])


# ---------------------------------------------------------------------------
# Seed helper
# ---------------------------------------------------------------------------

def test_gray_scott_seed_background_and_patch():
    """Background nodes sit near (u_bg=1, v_bg=0) up to the whole-field
    noise (see ``_gray_scott_seed``'s docstring for why the noise is
    whole-field, not confined to the seeded patch); seeded-patch nodes sit
    near (u_seed=0.5, v_seed=0.25). ``noise=0.03`` by default, so a 5-sigma
    band comfortably bounds both without being so loose the test can't catch
    a real regression (e.g. swapped seed/background values).
    """
    _, V = fem.build_mesh("unit_square", 12, degree=1)
    coords = fem.node_coords(V)
    u0, v0 = _gray_scott_seed(coords)

    dx = np.abs(coords[:, 0] - 0.5)
    dy = np.abs(coords[:, 1] - 0.5)
    seeded = (dx < 0.06) & (dy < 0.06)

    assert seeded.sum() > 0, "expected at least one seeded node at this resolution"
    band = 5 * 0.03
    assert np.all(np.abs(u0[~seeded] - 1.0) < band)
    assert np.all(np.abs(v0[~seeded] - 0.0) < band)
    assert np.all(np.abs(u0[seeded] - 0.5) < band)
    assert np.all(np.abs(v0[seeded] - 0.25) < band)
    # the seeded patch must be visibly offset from the background on average
    # (not just noise) -- this is what makes it a "seed" at all.
    assert u0[seeded].mean() < u0[~seeded].mean() - 0.2
    assert v0[seeded].mean() > v0[~seeded].mean() + 0.15


# ---------------------------------------------------------------------------
# Composed 3-process system: pattern-forming tendency + boundedness
# ---------------------------------------------------------------------------

# resolution=16, F=0.037, k=0.06 (composite defaults), dt=1.0: empirically
# var(V) grows from the seeded patch by ~2.3-2.6x over 300 ticks and plateaus
# (verified stable to at least 1000 ticks) without blowing up -- see this
# study's report for the full parameter/timing exploration. Comfortably
# below that observed ratio for a non-flaky margin.
FAST_RESOLUTION = 16
FAST_STEPS = 300
GROWTH_RATIO_MIN = 1.5


def _run_short(n_steps=FAST_STEPS, resolution=FAST_RESOLUTION):
    core = build_core()
    doc = turing_patterns(core, resolution=resolution, F=0.037, k=0.06, Du=2e-5, Dv=1e-5, dt=1.0)
    sim = Composite({"state": doc}, core=core)
    v0 = np.array(doc["stores"]["v"], dtype=float)
    sim.run(n_steps)
    u_final = np.array(sim.state["stores"]["u"], dtype=float)
    v_final = np.array(sim.state["stores"]["v"], dtype=float)
    return v0, u_final, v_final


def test_turing_patterns_heterogeneity_grows():
    """Headline composability proof: DiffusionProcess (x2) and
    GrayScottReactionProcess are wired together ONLY through shared bigraph
    stores (stores.u / stores.v / stores.source_u / stores.source_v) --
    none of the three processes knows about the other two.

    A short run from the seeded initial condition must grow V's spatial
    variance well above its initial (mostly-uniform-background) value; this
    can only happen if the reaction process's rate fields are actually
    reaching the diffusion processes and vice versa -- i.e. it is a direct
    behavioral proof of the composition, not just that the code runs.
    """
    v0, _u_final, v_final = _run_short()
    var0 = float(np.var(v0))
    var_final = float(np.var(v_final))
    assert var_final > GROWTH_RATIO_MIN * var0
    assert var0 > 0.0  # sanity: the seeded patch does perturb the initial variance


def test_turing_patterns_fields_stay_bounded():
    """No accumulate-vs-overwrite regression: source_u/source_v must be
    ``overwrite[array[float]]`` (fresh per-tick rates), not additive, or the
    coupled fields would blow up well past a physically-sane range instead
    of settling into a bounded pattern. Loose [0,1]-ish band (not exactly
    [0,1] -- discretization/reaction overshoot is expected and fine).
    """
    _v0, u_final, v_final = _run_short(n_steps=500)
    assert np.all(u_final > -0.5) and np.all(u_final < 1.5)
    assert np.all(v_final > -0.5) and np.all(v_final < 1.5)
    assert np.all(np.isfinite(u_final))
    assert np.all(np.isfinite(v_final))
