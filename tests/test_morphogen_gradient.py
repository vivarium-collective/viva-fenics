"""Fast tests for the morphogen-gradient showcase: ``DiffusionProcess`` (with
a Dirichlet source boundary c=c0 at x=0) ⊕ ``LinearDegradationProcess``
(``-k*c``), coupled purely through shared bigraph stores (see
``viva_fenics.composites.morphogen_gradient``).

These are all small-mesh / short-run checks -- the production study's
multi-regime, resolution=64/800-tick run (see
``studies/transient-diffusion/sims/run.py``) does not belong in the unit
test suite. What's tested here instead:

1. ``LinearDegradationProcess.update`` math is correct in isolation
   (source = -k*c, no double-application of ``interval``).
2. ``DiffusionProcess``'s new ``apply_boundary``/``boundary_value`` config
   actually pins the x=0 face via a real Dirichlet BC, and leaves every
   pre-existing (``apply_boundary=False``) call path bit-for-bit unchanged.
3. The composed 2-process system reaches a genuinely EXPONENTIAL steady
   gradient whose fitted decay length matches the analytic
   lambda=sqrt(D/k) -- the headline composability + physics claim -- on a
   small mesh / short run, real dolfinx throughout.
4. Fields stay bounded and non-negative (a degrading, boundary-sourced
   field should never go negative or blow up).
"""
from __future__ import annotations

import numpy as np
from process_bigraph import Composite

from viva_fenics.core import build_core
from viva_fenics.composites.morphogen_gradient import morphogen_gradient
from viva_fenics.processes.diffusion import DiffusionProcess
from viva_fenics.processes.reaction import LinearDegradationProcess
from viva_fenics import fem


def test_morphogen_gradient_generator_registered():
    from viva_superpowers.composite_generator import _REGISTRY
    assert any(e.endswith(".morphogen_gradient") for e in _REGISTRY)


def test_linear_degradation_process_registered_in_core():
    core = build_core()
    assert core.link_registry.get("LinearDegradationProcess") is not None


# ---------------------------------------------------------------------------
# LinearDegradationProcess.update math
# ---------------------------------------------------------------------------

def test_linear_degradation_source_is_minus_k_c():
    core = build_core()
    p = LinearDegradationProcess(config={"k": 2.0}, core=core)
    c = np.array([0.0, 0.5, 1.0, 2.0])
    out = p.update({"solution": c}, interval=0.01)
    assert np.allclose(out["source"], -2.0 * c)


def test_linear_degradation_zero_field_is_inert():
    core = build_core()
    p = LinearDegradationProcess(config={"k": 3.0}, core=core)
    c = np.zeros(10)
    out = p.update({"solution": c}, interval=0.01)
    assert np.allclose(out["source"], 0.0)


def test_linear_degradation_no_interval_scaling():
    """Pure RATE field -- no *interval, since DiffusionProcess multiplies
    "source" by dt internally (same convention as LogisticReactionProcess /
    GrayScottReactionProcess; see their tests of the same name)."""
    core = build_core()
    p = LinearDegradationProcess(config={"k": 1.5}, core=core)
    c = np.full(8, 0.4)
    out_small = p.update({"solution": c}, interval=0.001)
    out_large = p.update({"solution": c}, interval=1.0)
    assert np.allclose(out_small["source"], out_large["source"])
    assert np.allclose(out_small["source"], -1.5 * 0.4)


def test_linear_degradation_source_is_overwrite_not_additive():
    """Regression guard: outputs()["source"] must be
    overwrite[array[float]], or the shared stores.source would accumulate
    the instantaneous rate every tick instead of being replaced by the
    current -k*c reading."""
    core = build_core()
    p = LinearDegradationProcess(config={"k": 1.0}, core=core)
    assert p.outputs()["source"] == "overwrite[array[float]]"


# ---------------------------------------------------------------------------
# DiffusionProcess Dirichlet source-boundary option
# ---------------------------------------------------------------------------

def test_diffusion_boundary_off_is_unchanged():
    """apply_boundary=False (the default) must reproduce the exact
    pre-existing all-Neumann behavior: a step from a uniform field with zero
    source stays uniform (no boundary pinning anything down)."""
    core = build_core()
    p = DiffusionProcess(config={"resolution": 8, "D": 0.1, "dt": 0.01}, core=core)
    p._ensure_mesh()
    n = fem.node_coords(p._V).shape[0]
    c0 = np.full(n, 0.5)
    out = p.update({"solution": c0, "source": np.zeros(n)}, interval=0.01)
    new_field = c0 + np.array(out["solution"])
    assert np.allclose(new_field, 0.5, atol=1e-10)


def test_diffusion_boundary_on_pins_x0_face():
    """apply_boundary=True must pin every x=0 node to boundary_value after
    a step, regardless of the incoming field there."""
    core = build_core()
    p = DiffusionProcess(
        config={"resolution": 8, "D": 0.1, "dt": 0.01, "apply_boundary": True, "boundary_value": 0.7},
        core=core,
    )
    p._ensure_mesh()
    coords = fem.node_coords(p._V)
    n = coords.shape[0]
    c0 = np.zeros(n)
    out = p.update({"solution": c0, "source": np.zeros(n)}, interval=0.01)
    new_field = c0 + np.array(out["solution"])
    on_boundary = np.isclose(coords[:, 0], 0.0)
    assert on_boundary.sum() > 0
    assert np.allclose(new_field[on_boundary], 0.7, atol=1e-10)


# ---------------------------------------------------------------------------
# Composed 2-process system: exponential steady gradient
# ---------------------------------------------------------------------------

# resolution=16, D=0.1, k=1.0 (lambda=sqrt(0.1/1.0)=0.3162), dt=0.05, 120
# ticks (t=6.0): empirically (see study report) the fitted decay length is
# already within ~2% of the analytic value by this point and stable to 4
# significant figures well before it -- comfortably below the 15%
# non-flaky margin used here (small-mesh dolfinx solves carry more
# discretization noise than the production resolution=64 run).
FAST_RESOLUTION = 16
FAST_DT = 0.05
FAST_STEPS = 120
FAST_D = 0.1
FAST_K = 1.0
FAST_C0 = 1.0
LAMBDA_REL_ERR_MAX = 0.15


def _run_to_steady(resolution=FAST_RESOLUTION, dt=FAST_DT, n_steps=FAST_STEPS, D=FAST_D, k=FAST_K, c0=FAST_C0):
    core = build_core()
    doc = morphogen_gradient(core, D=D, k=k, c0=c0, resolution=resolution, dt=dt)
    sim = Composite({"state": doc}, core=core)
    sim.run(n_steps * dt)

    _, V = fem.build_mesh("unit_square", resolution, degree=1)
    coords = fem.node_coords(V)
    c = np.array(sim.state["stores"]["solution"], dtype=float)
    return coords, c


def test_morphogen_gradient_source_boundary_reaches_c0():
    """Headline composability wiring check: the Dirichlet BC baked into
    DiffusionProcess's config by the composite must actually take effect
    through the real Composite tick loop -- x=0 nodes must sit at c0."""
    coords, c = _run_to_steady()
    x = coords[:, 0]
    on_boundary = np.isclose(x, 0.0)
    assert on_boundary.sum() > 0
    assert np.allclose(c[on_boundary], FAST_C0, atol=1e-6)


def test_morphogen_gradient_is_exponential_with_correct_decay_length():
    """Headline physics claim: DiffusionProcess and LinearDegradationProcess
    are wired together ONLY through shared bigraph stores (stores.solution /
    stores.source) -- neither knows about the other, or about the SDD model
    as a system. A real Composite run must nonetheless produce a genuinely
    exponential steady gradient whose fitted decay length matches the
    analytic lambda=sqrt(D/k) -- this can only happen if both the source
    boundary (baked into DiffusionProcess's config) and the degradation
    term (flowing through stores.source) are actually taking effect.
    """
    coords, c = _run_to_steady()
    x = coords[:, 0]
    lam = np.sqrt(FAST_D / FAST_K)

    # fit region: past a couple of mesh cells from the source, short of the
    # far no-flux boundary's influence (see sims/run.py's module docstring
    # for the empirical justification of this window).
    mask = (x > 2.0 / FAST_RESOLUTION) & (x < 0.4) & (c > 1e-9)
    assert mask.sum() > 10, "expected a well-populated fit region at this resolution"
    slope, _intercept = np.polyfit(x[mask], np.log(c[mask]), 1)
    fitted_lambda = -1.0 / slope

    rel_err = abs(fitted_lambda - lam) / lam
    assert rel_err < LAMBDA_REL_ERR_MAX, (
        f"fitted lambda={fitted_lambda:.4f} vs analytic lambda={lam:.4f} "
        f"(rel_err={rel_err:.2%}) -- expected < {LAMBDA_REL_ERR_MAX:.0%}"
    )


def test_morphogen_gradient_fields_bounded_and_nonnegative():
    """No accumulate-vs-overwrite regression: stores.source must be
    overwrite[array[float]] (a fresh -k*c reading each tick), not additive,
    or the coupled field would blow up or go unboundedly negative instead
    of settling into a bounded, non-negative gradient between 0 and c0."""
    _coords, c = _run_to_steady(n_steps=200)
    assert np.all(np.isfinite(c))
    assert np.all(c >= -1e-6)
    assert np.all(c <= FAST_C0 * 1.05)


def test_morphogen_gradient_decay_length_shrinks_with_k():
    """lambda=sqrt(D/k) must control the gradient's range: a higher k
    (faster degradation) must produce a visibly SHORTER decay length --
    the same-D, different-k comparison the study's variants sweep."""
    core = build_core()
    D, c0, resolution, dt, n_steps = 0.1, 1.0, FAST_RESOLUTION, FAST_DT, FAST_STEPS

    def _fitted_lambda(k):
        doc = morphogen_gradient(core, D=D, k=k, c0=c0, resolution=resolution, dt=dt)
        sim = Composite({"state": doc}, core=core)
        sim.run(n_steps * dt)
        _, V = fem.build_mesh("unit_square", resolution, degree=1)
        coords = fem.node_coords(V)
        x = coords[:, 0]
        c = np.array(sim.state["stores"]["solution"], dtype=float)
        mask = (x > 2.0 / resolution) & (x < 0.4) & (c > 1e-9)
        slope, _ = np.polyfit(x[mask], np.log(c[mask]), 1)
        return -1.0 / slope

    lambda_k1 = _fitted_lambda(1.0)
    lambda_k4 = _fitted_lambda(4.0)
    assert lambda_k4 < lambda_k1
    # analytic ratio is sqrt(4)=2x; require at least a 1.5x measured shrink
    # (headroom below the analytic value for a non-flaky small-mesh check).
    assert lambda_k1 / lambda_k4 > 1.5
