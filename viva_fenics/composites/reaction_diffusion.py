"""Composite generator coupling ``DiffusionProcess`` (Task 3) with
``LogisticReactionProcess`` (Task 4) through shared bigraph stores to produce
Fisher-KPP dynamics (``du/dt = D*laplacian(u) + r*u*(1 - u/K)``) BY
COMPOSITION -- neither process implements Fisher-KPP itself, nor calls into
the other. The coupling is entirely the document wiring below:

- ``stores.solution`` is read+written by ``DiffusionProcess`` (in/out) and
  read by ``LogisticReactionProcess`` (in). Both processes see the same
  current field each tick.
- ``stores.source`` is written by ``LogisticReactionProcess`` (out,
  ``overwrite[array[float]]`` -- a fresh per-tick rate, not an accumulated
  delta) and read by ``DiffusionProcess`` (in, which multiplies it by dt
  internally as its reaction/source term).
- ``stores.integral`` is written by ``DiffusionProcess`` (out,
  ``overwrite[float]``) as a mass-bookkeeping sensor reading.

This is the composability showcase: swap ``LogisticReactionProcess`` for any
other process that reads "solution" and writes "source" and the diffusion
process is none the wiser.
"""

from __future__ import annotations

from viva_superpowers.composite_generator import composite_generator

from viva_fenics import fem
from viva_fenics.processes.diffusion import _gaussian_bump


@composite_generator(
    name="reaction_diffusion",
    description="Fisher-KPP reaction-diffusion via DiffusionProcess ⊕ LogisticReactionProcess composition.",
    parameters={
        "resolution": {"type": "integer", "default": 32,
                        "description": "Mesh cells per side of the unit square"},
        "D": {"type": "float", "default": 0.05,
               "description": "Diffusion coefficient"},
        "r": {"type": "float", "default": 1.0,
               "description": "Logistic growth rate"},
        "dt": {"type": "float", "default": 0.01,
                "description": "Backward-Euler timestep size (shared by both processes)"},
    },
)
def reaction_diffusion(core=None, *, resolution=32, D=0.05, r=1.0, dt=0.01):
    # Seed stores.solution / stores.source with real, correctly-shaped nodal
    # arrays (NOT `[]`) -- see transient_diffusion's docstring for why an
    # empty starting array corrupts the first apply (array[float]'s "was
    # initialized empty -> replace" special-case).
    _, V = fem.build_mesh("unit_square", resolution, degree=1)
    coords = fem.node_coords(V)
    bump = _gaussian_bump(coords)

    return {
        "diffusion": {
            "_type": "process",
            "address": "local:DiffusionProcess",
            "config": {"resolution": resolution, "D": D, "dt": dt},
            "inputs": {
                "source": ["stores", "source"],
                "solution": ["stores", "solution"],
            },
            "outputs": {
                "solution": ["stores", "solution"],
                "integral": ["stores", "integral"],
            },
            "interval": dt,
        },
        "reaction": {
            "_type": "process",
            "address": "local:LogisticReactionProcess",
            "config": {"r": r},
            "inputs": {
                "solution": ["stores", "solution"],
            },
            "outputs": {
                "source": ["stores", "source"],
            },
            "interval": dt,
        },
        "stores": {
            "solution": bump.tolist(),
            "source": [0.0] * len(bump),
            "integral": 0.0,
        },
        "emitter": {
            "_type": "step",
            "address": "local:RAMEmitter",
            "config": {"emit": {"solution": "array[float]", "integral": "float"}},
            "inputs": {
                "solution": ["stores", "solution"],
                "integral": ["stores", "integral"],
            },
        },
    }
