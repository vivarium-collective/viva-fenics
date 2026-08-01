"""Composite generator coupling TWO ``DiffusionProcess`` instances (one per
species, differential diffusivities ``Du`` > ``Dv``) with ONE
``GrayScottReactionProcess`` through shared bigraph stores, producing the
Gray-Scott reaction-diffusion system:

    dU/dt = Du*laplacian(U) - U*V^2 + F*(1-U)
    dV/dt = Dv*laplacian(V) + U*V^2 - (F+k)*V

BY COMPOSITION -- none of the three processes implements Gray-Scott itself,
and none calls into either of the others. The coupling is entirely the
document wiring below:

- ``stores.u`` is read+written by ``diffusion_u`` (in/out, its "solution"
  port) and read by ``reaction`` (in, its "u" port). All two see the same
  current U field each tick.
- ``stores.v`` is read+written by ``diffusion_v`` (in/out, its "solution"
  port) and read by ``reaction`` (in, its "v" port).
- ``stores.source_u`` / ``stores.source_v`` are written by ``reaction`` (out,
  ``overwrite[array[float]]`` -- fresh per-tick rates, not accumulated
  deltas) and read by ``diffusion_u`` / ``diffusion_v`` respectively (in,
  each multiplying its own species' rate by dt internally as its reaction
  term).
- ``stores.integral_u`` / ``stores.integral_v`` are written by
  ``diffusion_u`` / ``diffusion_v`` (out, ``overwrite[float]``) as
  mass-bookkeeping sensor readings.

This is the investigation's headline composability showcase: three
independently-authored processes -- two copies of the SAME
``DiffusionProcess`` class (parameterized only by which species' stores
they're wired to and their own diffusivity) plus one
``GrayScottReactionProcess`` -- produce a genuine 2D Turing instability
(spontaneous spot/stripe/labyrinth self-organization from a near-uniform
initial condition) purely through shared-store wiring. Swap
``GrayScottReactionProcess`` for any other process that reads "u"/"v" and
writes "source_u"/"source_v" and the two diffusion processes are none the
wiser -- exactly the substitutability property ``reaction_diffusion``
(Fisher-KPP, one species) already demonstrates, extended here to a
two-species system whose only way to pattern-form AT ALL is the coupling
between three processes, not two.
"""

from __future__ import annotations

import numpy as np
from viva_superpowers.composite_generator import composite_generator

from viva_fenics import fem


def _gray_scott_seed(
    coords,
    *,
    seed_size=0.12,
    center=(0.5, 0.5),
    u_bg=1.0,
    v_bg=0.0,
    u_seed=0.5,
    v_seed=0.25,
    noise=0.03,
    rng_seed=0,
):
    """Nodal (U, V) initial condition for Gray-Scott: a uniform background
    (U=u_bg, V=v_bg) everywhere EXCEPT a small central square perturbed to
    (U=u_seed, V=v_seed), plus tiny noise -- the infinitesimal
    symmetry-breaking perturbation a real Turing instability amplifies.
    Deterministic (fixed ``rng_seed``) so the composite's initial state is
    reproducible across runs/resolutions.

    The noise is added to EVERY node, not just the seeded patch (unlike an
    earlier version of this function). Empirically (see this composite's
    study report for the calibration sweep): with THIS module's
    Du/Dv/dt/domain-size combination, a single localized seed patch alone --
    even a large one, even with noise confined to it -- diffuses below the
    local UV^2 autocatalytic threshold before the (deliberately slow, F~0.03-
    0.06) reaction can win, and decays straight back to the trivial (U=1,
    V=0) fixed point instead of nucleating a pattern. Scattering the same
    tiny noise across the whole domain gives many simultaneous
    below-threshold perturbations; ones that happen to exceed the local
    growth threshold (governed by F/k, not location) nucleate independently
    across the domain, which is what actually produces a genuine, tested,
    non-decaying Turing pattern for this study's regimes. This is standard
    practice for Gray-Scott initial conditions in the wider literature/demo
    ecosystem, not a shortcut around the physics -- the PDE and its
    coefficients are unchanged; only the initial perturbation's spatial
    extent differs from a single point.
    """
    coords = np.asarray(coords, dtype=float)
    rng = np.random.default_rng(rng_seed)
    n = coords.shape[0]

    dx = np.abs(coords[:, 0] - center[0])
    dy = np.abs(coords[:, 1] - center[1])
    seeded = (dx < seed_size / 2) & (dy < seed_size / 2)

    u = np.full(n, u_bg, dtype=float)
    v = np.full(n, v_bg, dtype=float)
    u[seeded] = u_seed
    v[seeded] = v_seed

    u += rng.normal(scale=noise, size=n)
    v += rng.normal(scale=noise, size=n)
    np.clip(u, 0.0, 1.0, out=u)
    np.clip(v, 0.0, 1.0, out=v)
    return u, v


@composite_generator(
    name="turing_patterns",
    description=(
        "Gray-Scott Turing pattern formation: two DiffusionProcess "
        "instances (species U, V; Du != Dv) ⊕ one GrayScottReactionProcess "
        "composed through shared stores."
    ),
    parameters={
        "resolution": {"type": "integer", "default": 128,
                        "description": "Mesh cells per side of the unit square"},
        "F": {"type": "float", "default": 0.037,
               "description": "Gray-Scott feed rate"},
        "k": {"type": "float", "default": 0.06,
               "description": "Gray-Scott kill rate"},
        "Du": {"type": "float", "default": 2e-5,
                "description": "Diffusion coefficient of species U (activator substrate)"},
        "Dv": {"type": "float", "default": 1e-5,
                "description": "Diffusion coefficient of species V (~half of Du -- differential diffusion drives the Turing instability)"},
        "dt": {"type": "float", "default": 1.0,
                "description": "Backward-Euler timestep size (shared by both diffusion processes and the reaction)"},
    },
)
def turing_patterns(core=None, *, F=0.037, k=0.06, Du=2e-5, Dv=1e-5, resolution=128, dt=1.0):
    # Seed stores.u / stores.v / stores.source_u / stores.source_v with real,
    # correctly-shaped nodal arrays (NOT `[]`) -- see reaction_diffusion's
    # docstring for why an empty starting array corrupts the first apply of
    # an array[float]/overwrite[array[float]] store.
    _, V = fem.build_mesh("unit_square", resolution, degree=1)
    coords = fem.node_coords(V)
    u0, v0 = _gray_scott_seed(coords)
    n = len(u0)

    return {
        "diffusion_u": {
            "_type": "process",
            "address": "local:DiffusionProcess",
            "config": {"resolution": resolution, "D": Du, "dt": dt},
            "inputs": {
                "source": ["stores", "source_u"],
                "solution": ["stores", "u"],
            },
            "outputs": {
                "solution": ["stores", "u"],
                "integral": ["stores", "integral_u"],
            },
            "interval": dt,
        },
        "diffusion_v": {
            "_type": "process",
            "address": "local:DiffusionProcess",
            "config": {"resolution": resolution, "D": Dv, "dt": dt},
            "inputs": {
                "source": ["stores", "source_v"],
                "solution": ["stores", "v"],
            },
            "outputs": {
                "solution": ["stores", "v"],
                "integral": ["stores", "integral_v"],
            },
            "interval": dt,
        },
        "reaction": {
            "_type": "process",
            "address": "local:GrayScottReactionProcess",
            "config": {"F": F, "k": k},
            "inputs": {
                "u": ["stores", "u"],
                "v": ["stores", "v"],
            },
            "outputs": {
                "source_u": ["stores", "source_u"],
                "source_v": ["stores", "source_v"],
            },
            "interval": dt,
        },
        "stores": {
            "u": u0.tolist(),
            "v": v0.tolist(),
            "source_u": [0.0] * n,
            "source_v": [0.0] * n,
            "integral_u": 0.0,
            "integral_v": 0.0,
        },
        "emitter": {
            "_type": "step",
            "address": "local:RAMEmitter",
            "config": {"emit": {
                "u": "array[float]",
                "v": "array[float]",
                "integral_u": "float",
                "integral_v": "float",
            }},
            "inputs": {
                "u": ["stores", "u"],
                "v": ["stores", "v"],
                "integral_u": ["stores", "integral_u"],
                "integral_v": ["stores", "integral_v"],
            },
        },
    }
