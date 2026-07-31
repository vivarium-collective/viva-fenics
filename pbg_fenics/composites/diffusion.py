"""Composite generator for a standalone transient diffusion run.

Wires a single ``DiffusionProcess`` (Task 3) to a shared ``solution``/``source``
store pair and a RAM emitter, so the dashboard's Composites tab / a bare
workspace run can build and execute it. ``source`` starts as a zeros store
(no external source in the standalone case); a later composite (Task 4/5)
can wire a sibling process into the same ``stores.source`` for coupling.
"""

from __future__ import annotations

from viva_superpowers.composite_generator import composite_generator

from pbg_fenics import fem
from pbg_fenics.processes.diffusion import _gaussian_bump


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
    # Seed ``stores.solution``/``stores.source`` with real, correctly-shaped
    # nodal arrays (NOT ``[]``). ``array[float]`` apply special-cases a
    # truly-empty starting array by *replacing* it with the first update
    # instead of adding (see bigraph_schema/methods/apply.py's `Array`
    # dispatch: "State was initialized empty -- replace with update"). Since
    # DiffusionProcess reads "solution" back as an input every tick, an
    # empty placeholder would silently corrupt the very first tick's apply
    # (the store would become the raw delta instead of prev + delta). Poisson
    # Task 2's Step never reads its own output back, so it never hit this.
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
