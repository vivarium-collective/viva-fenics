"""Composite generator coupling :class:`AdvectionDiffusionProcess` with
:class:`LogisticReactionProcess` through shared bigraph stores to produce an
**advection-diffusion-reaction** field BY COMPOSITION -- the vent / pore
reactive-transport case of the viva-genesis origins-of-life campaign (full
modeling plan section 4.5, the og-00 vent reactor's "advection + reaction").

Same composition pattern as ``reaction_diffusion`` (``DiffusionProcess`` +
``LogisticReactionProcess``), but the transport process now carries a Darcy
upflow (``velocity_mode="stream_function"``, a uniform +y velocity), so the
reacting solute is advected up the column as it diffuses and grows -- neither
process knows about advection or the other; the coupling is entirely the
document wiring below. Swap ``AdvectionDiffusionProcess`` back for
``DiffusionProcess`` (or set ``velocity_y=0``) and the composite reproduces
plain Fisher-KPP.
"""

from __future__ import annotations

from viva_superpowers.composite_generator import composite_generator

from viva_fenics import fem
from viva_fenics.processes.diffusion import _gaussian_bump


@composite_generator(
    name="reactive_transport",
    description=(
        "Advection-diffusion-reaction via AdvectionDiffusionProcess ⊕ "
        "LogisticReactionProcess composition: a reacting solute advected up a "
        "Darcy upflow while it diffuses and grows (the vent/pore reactive-"
        "transport case). Real dolfinx FEM."
    ),
    parameters={
        "resolution": {"type": "integer", "default": 32,
                       "description": "Mesh cells per side of the unit square"},
        "D": {"type": "float", "default": 0.05,
              "description": "Diffusion coefficient"},
        "velocity_y": {"type": "float", "default": 0.4,
                       "description": "Darcy upflow speed (+y advection)"},
        "r": {"type": "float", "default": 1.0, "description": "Logistic growth rate"},
        "dt": {"type": "float", "default": 0.01,
               "description": "Backward-Euler timestep (shared by both processes)"},
    },
)
def reactive_transport(core=None, *, resolution=32, D=0.05, velocity_y=0.4, r=1.0, dt=0.01):
    _, V = fem.build_mesh("unit_square", resolution, degree=1)
    coords = fem.node_coords(V)
    bump = _gaussian_bump(coords, center=(0.5, 0.25))  # seed near the base; upflow carries it up

    return {
        "transport": {
            "_type": "process",
            "address": "local:AdvectionDiffusionProcess",
            "config": {"resolution": resolution, "D": D, "dt": dt,
                       "velocity_mode": "stream_function",
                       "velocity_x": 0.0, "velocity_y": velocity_y},
            "inputs": {
                "source": ["stores", "source"],
                "solution": ["stores", "solution"],
                "velocity": ["stores", "velocity"],
                "temperature": ["stores", "temperature"],
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
            "inputs": {"solution": ["stores", "solution"]},
            "outputs": {"source": ["stores", "source"]},
            "interval": dt,
        },
        "stores": {
            "solution": bump.tolist(),
            "source": [0.0] * len(bump),
            "velocity": [0.0] * (2 * len(bump)),
            "temperature": [0.0] * len(bump),
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
