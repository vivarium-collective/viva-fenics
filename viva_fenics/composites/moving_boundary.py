"""Composite generator for a standalone moving-boundary (ALE) run.

Wires a single ``MovingBoundaryProcess`` (real dolfinx diffusion re-solved on
a prescribed-ALE deforming mesh -- see
``viva_fenics.processes.moving_boundary``'s module docstring) to a RAM
emitter so the dashboard's Composites tab / a bare workspace run can build
and execute it. ``boundary_position``/``domain_measure`` are ``overwrite``
sensor outputs the process alone writes, so their store entries only need
correctly-shaped placeholders; ``solution`` is seeded with a real nodal
gaussian bump (never ``[]`` -- see ``reaction_diffusion``'s docstring for
why an empty starting array corrupts the first ``array[float]`` apply).
"""

from __future__ import annotations

from viva_superpowers.composite_generator import composite_generator

from viva_fenics.processes.diffusion import _gaussian_bump
from viva_fenics.processes.moving_boundary import _build_mesh, reference_coords


@composite_generator(
    name="moving_boundary",
    description="Diffusion on a deforming (ALE) domain with a prescribed moving boundary.",
    parameters={
        "resolution": {"type": "integer", "default": 32,
                        "description": "Mesh cells per side of the reference unit square"},
        "amplitude": {"type": "number", "default": 0.3,
                       "description": "Boundary oscillation amplitude, or growth rate for mode='grow'"},
        "omega": {"type": "number", "default": 3.14159,
                   "description": "Angular frequency of the boundary oscillation (mode='oscillate')"},
        "dt": {"type": "number", "default": 0.01,
                "description": "Backward-Euler substep size, also the mesh-motion substep size"},
        "mode": {"type": "string", "default": "oscillate",
                  "description": "Boundary motion law: 'oscillate' or 'grow'"},
    },
)
def moving_boundary(core=None, *, resolution=32, amplitude=0.3, omega=3.14159, dt=0.01, mode="oscillate"):
    _, V, _y_ref = _build_mesh(resolution, degree=1)
    coords = reference_coords(V)
    bump = _gaussian_bump(coords)

    return {
        "moving_boundary": {
            "_type": "process",
            "address": "local:MovingBoundaryProcess",
            "config": {
                "resolution": resolution,
                "amplitude": amplitude,
                "omega": omega,
                "dt": dt,
                "mode": mode,
            },
            "inputs": {
                "solution": ["stores", "solution"],
                "boundary_drive": ["stores", "boundary_drive"],
            },
            "outputs": {
                "solution": ["stores", "solution"],
                "boundary_position": ["stores", "boundary_position"],
                "domain_measure": ["stores", "domain_measure"],
            },
            "interval": dt,
        },
        "stores": {
            "solution": bump.tolist(),
            "boundary_drive": 0.0,
            "boundary_position": 1.0,
            "domain_measure": 1.0,
        },
        "emitter": {
            "_type": "step",
            "address": "local:RAMEmitter",
            "config": {
                "emit": {
                    "solution": "array[float]",
                    "boundary_position": "float",
                    "domain_measure": "float",
                },
            },
            "inputs": {
                "solution": ["stores", "solution"],
                "boundary_position": ["stores", "boundary_position"],
                "domain_measure": ["stores", "domain_measure"],
            },
        },
    }
