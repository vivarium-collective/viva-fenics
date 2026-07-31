"""Composite generator for a standalone transient diffusion run.

Wires a single ``DiffusionProcess`` (Task 3) to a shared ``solution``/``source``
store pair and a RAM emitter, so the dashboard's Composites tab / a bare
workspace run can build and execute it. ``source`` starts as a zeros store
(no external source in the standalone case); a later composite (Task 4/5)
can wire a sibling process into the same ``stores.source`` for coupling.
"""

from __future__ import annotations

from viva_superpowers.composite_generator import composite_generator


@composite_generator(
    name="transient_diffusion",
    description="Transient (backward-Euler) diffusion of a gaussian bump on the unit square.",
    parameters={
        "resolution": {"type": "integer", "default": 32,
                        "description": "Mesh cells per side of the unit square"},
        "D": {"type": "float", "default": 0.1,
               "description": "Diffusion coefficient"},
        "dt": {"type": "float", "default": 0.01,
                "description": "Backward-Euler timestep size"},
    },
)
def transient_diffusion(core=None, *, resolution=32, D=0.1, dt=0.01):
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
        },
        "stores": {
            "solution": [],
            "source": [],
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
