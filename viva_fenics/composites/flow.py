"""Composite generator for a standalone lid-driven-cavity Navier-Stokes run.

Wires a single ``NavierStokesProcess`` (real dolfinx IPCS solve) to a RAM
emitter so the dashboard's Composites tab / a bare workspace run can build
and execute it. ``velocity_x``/``velocity_y``/``pressure``/``speed_integral``
are all ``overwrite`` outputs the process alone writes (see
``NavierStokesProcess.outputs`` for why), so the store block only needs
correctly-shaped zero-valued placeholders -- never ``[]``, which corrupts
the first ``overwrite`` apply (see ``reaction_diffusion``'s docstring for
that lesson).
"""

from __future__ import annotations

from viva_superpowers.composite_generator import composite_generator

from viva_fenics.processes.flow import zero_fields


@composite_generator(
    name="navier_stokes",
    description="Incompressible lid-driven cavity flow (IPCS) -- velocity + pressure fields.",
    parameters={
        "reynolds": {"type": "number", "default": 100.0,
                      "description": "Reynolds number (lid_velocity * L / nu)"},
        "resolution": {"type": "integer", "default": 32,
                        "description": "Mesh cells per side of the unit-square cavity"},
        "dt": {"type": "number", "default": 0.01,
                "description": "IPCS substep timestep size"},
    },
)
def navier_stokes(core=None, *, reynolds=100.0, resolution=32, dt=0.01):
    velocity_x0, velocity_y0, pressure0 = zero_fields(resolution)
    body_force0 = [0.0] * (2 * len(velocity_x0))

    return {
        "flow": {
            "_type": "process",
            "address": "local:NavierStokesProcess",
            "config": {
                "resolution": resolution,
                "reynolds": reynolds,
                "dt": dt,
            },
            "inputs": {
                "body_force": ["stores", "body_force"],
            },
            "outputs": {
                "velocity_x": ["stores", "velocity_x"],
                "velocity_y": ["stores", "velocity_y"],
                "pressure": ["stores", "pressure"],
                "speed_integral": ["stores", "speed_integral"],
            },
            "interval": dt,
        },
        "stores": {
            "body_force": body_force0,
            "velocity_x": velocity_x0.tolist(),
            "velocity_y": velocity_y0.tolist(),
            "pressure": pressure0.tolist(),
            "speed_integral": 0.0,
        },
        "emitter": {
            "_type": "step",
            "address": "local:RAMEmitter",
            "config": {
                "emit": {
                    "velocity_x": "array[float]",
                    "velocity_y": "array[float]",
                    "pressure": "array[float]",
                    "speed_integral": "float",
                },
            },
            "inputs": {
                "velocity_x": ["stores", "velocity_x"],
                "velocity_y": ["stores", "velocity_y"],
                "pressure": ["stores", "pressure"],
                "speed_integral": ["stores", "speed_integral"],
            },
        },
    }
