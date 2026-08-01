"""Composite generator coupling ``DiffusionProcess`` (with a real Dirichlet
source boundary) with ``LinearDegradationProcess`` through shared bigraph
stores to produce the Source-Diffusion-Degradation (SDD) morphogen-gradient
model BY COMPOSITION -- neither process implements SDD itself, nor calls
into the other. The coupling is entirely the document wiring below:

- ``stores.solution`` (the morphogen concentration c) is read+written by
  ``DiffusionProcess`` (in/out, pinned to ``c0`` on the x=0 face via its
  ``apply_boundary``/``boundary_value`` config -- the morphogen production
  boundary) and read by ``LinearDegradationProcess`` (in).
- ``stores.source`` is written by ``LinearDegradationProcess`` (out,
  ``overwrite[array[float]]`` -- a fresh per-tick rate ``-k*c``, not an
  accumulated delta) and read by ``DiffusionProcess`` (in, which multiplies
  it by dt internally as its degradation term).
- ``stores.integral`` is written by ``DiffusionProcess`` (out,
  ``overwrite[float]``) as a mass-bookkeeping sensor reading.

Governing PDE (emergent, not implemented anywhere as a single equation):

    dc/dt = D*laplacian(c) - k*c,   c(x=0) = c0,   no-flux elsewhere

Steady state: c(x) = c0*exp(-x/lambda), decay length lambda = sqrt(D/k) --
the classic Wolpert "French flag" positional-information gradient. Every
other face of the unit square stays natural Neumann (zero-flux, the
DiffusionProcess default), and the domain is translationally symmetric in y
(the BC and initial condition depend only on x), so the field stays
effectively 1D throughout the run.

This is the same composability showcase as ``reaction_diffusion`` (Fisher-
KPP): swap ``LinearDegradationProcess`` for any other process that reads
"solution" and writes "source" and ``DiffusionProcess`` is none the wiser.
"""

from __future__ import annotations

import numpy as np
from viva_superpowers.composite_generator import composite_generator

from viva_fenics import fem


@composite_generator(
    name="morphogen_gradient",
    description=(
        "Source-Diffusion-Degradation (SDD) morphogen gradient: "
        "DiffusionProcess (Dirichlet source boundary c=c0) ⊕ "
        "LinearDegradationProcess composition; steady state "
        "c(x)=c0*exp(-x/lambda), lambda=sqrt(D/k)."
    ),
    parameters={
        "D": {"type": "float", "default": 0.1,
               "description": "Diffusion coefficient"},
        "k": {"type": "float", "default": 1.0,
               "description": "First-order degradation rate"},
        "c0": {"type": "float", "default": 1.0,
                "description": "Morphogen concentration at the source boundary (x=0)"},
        "resolution": {"type": "integer", "default": 64,
                        "description": "Mesh cells per side of the unit square"},
        "dt": {"type": "float", "default": 0.01,
                "description": "Backward-Euler timestep size (shared by both processes)"},
    },
)
def morphogen_gradient(core=None, *, D=0.1, k=1.0, c0=1.0, resolution=64, dt=0.01):
    # Seed stores.solution / stores.source with real, correctly-shaped nodal
    # arrays (NOT `[]`) -- see transient_diffusion's docstring for why an
    # empty starting array corrupts the first apply (array[float]'s "was
    # initialized empty -> replace" special-case). No morphogen has been
    # produced yet, so the field starts at zero everywhere (the source
    # boundary's Dirichlet BC pins x=0 to c0 starting from the very first
    # step, regardless of this seed).
    _, V = fem.build_mesh("unit_square", resolution, degree=1)
    coords = fem.node_coords(V)
    n = coords.shape[0]
    zeros = np.zeros(n)

    return {
        "diffusion": {
            "_type": "process",
            "address": "local:DiffusionProcess",
            "config": {
                "resolution": resolution, "D": D, "dt": dt,
                "apply_boundary": True, "boundary_value": c0,
            },
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
        "degradation": {
            "_type": "process",
            "address": "local:LinearDegradationProcess",
            "config": {"k": k},
            "inputs": {
                "solution": ["stores", "solution"],
            },
            "outputs": {
                "source": ["stores", "source"],
            },
            "interval": dt,
        },
        "stores": {
            "solution": zeros.tolist(),
            "source": zeros.tolist(),
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
