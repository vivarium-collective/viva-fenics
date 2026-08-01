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

from viva_fenics.processes.flow import (
    zero_fields,
    zero_channel_fields,
    POROUS_PRESSURE_DROP_DEFAULT,
    POROUS_VISCOSITY_DEFAULT,
)


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


@composite_generator(
    name="vortex_street",
    description=(
        "DFG 2D-2 benchmark -- incompressible flow past an off-center "
        "cylinder in a channel (real dolfinx IPCS on a gmsh-generated, "
        "cylinder-refined mesh); at Re=100 the wake sheds a von Karman "
        "vortex street with oscillating drag/lift."
    ),
    parameters={
        "reynolds": {"type": "number", "default": 100.0,
                      "description": "Reynolds number (U_mean * cylinder_diameter / nu)"},
        "dt": {"type": "number", "default": 0.0005,
                "description": (
                    "IPCS substep timestep size -- 0.0005, not the 0.001-0.002 "
                    "that suffices for the lid-driven-cavity study: at this "
                    "cylinder-refined mesh + Re=100, dt=0.001 (and even a "
                    "skew-symmetric convection form at dt=0.001) went unstable "
                    "(NaN) partway through the startup transient (~t=0.5s, "
                    "confirmed empirically during development); dt=0.0005 does not."
                )},
        "h_cylinder": {"type": "number", "default": 0.008,
                        "description": "Mesh element size at the cylinder boundary (accuracy driver)"},
        "h_far": {"type": "number", "default": 0.05,
                   "description": "Mesh element size far from the cylinder (channel inlet/outlet/walls)"},
    },
)
def vortex_street(core=None, *, reynolds=100.0, dt=0.0005, h_cylinder=0.008, h_far=0.05):
    velocity_x0, velocity_y0, pressure0, vorticity0 = zero_channel_fields(h_cylinder, h_far)

    return {
        "flow": {
            "_type": "process",
            "address": "local:CylinderFlowProcess",
            "config": {
                "h_cylinder": h_cylinder,
                "h_far": h_far,
                "reynolds": reynolds,
                "dt": dt,
            },
            "inputs": {
                "inflow_perturbation": ["stores", "inflow_perturbation"],
            },
            "outputs": {
                "velocity_x": ["stores", "velocity_x"],
                "velocity_y": ["stores", "velocity_y"],
                "pressure": ["stores", "pressure"],
                "vorticity": ["stores", "vorticity"],
                "drag_coeff": ["stores", "drag_coeff"],
                "lift_coeff": ["stores", "lift_coeff"],
                "elapsed_time": ["stores", "elapsed_time"],
            },
            "interval": dt,
        },
        "stores": {
            "inflow_perturbation": 0.0,
            "velocity_x": velocity_x0.tolist(),
            "velocity_y": velocity_y0.tolist(),
            "pressure": pressure0.tolist(),
            "vorticity": vorticity0.tolist(),
            "drag_coeff": 0.0,
            "lift_coeff": 0.0,
            "elapsed_time": 0.0,
        },
        "emitter": {
            "_type": "step",
            "address": "local:RAMEmitter",
            "config": {
                "emit": {
                    "velocity_x": "array[float]",
                    "velocity_y": "array[float]",
                    "pressure": "array[float]",
                    "vorticity": "array[float]",
                    "drag_coeff": "float",
                    "lift_coeff": "float",
                    "elapsed_time": "float",
                },
            },
            "inputs": {
                "velocity_x": ["stores", "velocity_x"],
                "velocity_y": ["stores", "velocity_y"],
                "pressure": ["stores", "pressure"],
                "vorticity": ["stores", "vorticity"],
                "drag_coeff": ["stores", "drag_coeff"],
                "lift_coeff": ["stores", "lift_coeff"],
                "elapsed_time": ["stores", "elapsed_time"],
            },
        },
    }


@composite_generator(
    name="porous_lattice",
    description=(
        "Steady Stokes flow through a periodic circular-pillar lattice "
        "(real Taylor-Hood mixed dolfinx solve on a gmsh-generated porous "
        "microstructure) -- computes the effective (Darcy) permeability "
        "k_eff of the medium."
    ),
    parameters={
        "nx": {"type": "integer", "default": 4, "description": "Pillar grid columns"},
        "ny": {"type": "integer", "default": 4, "description": "Pillar grid rows"},
        "pillar_radius": {"type": "number", "default": 0.08,
                            "description": "Pillar disk radius (porosity-control knob; must be < half the lattice spacing)"},
        "h_pillar": {"type": "number", "default": 0.015,
                      "description": "Mesh element size at every pillar boundary (accuracy driver)"},
        "h_far": {"type": "number", "default": 0.05,
                   "description": "Mesh element size away from the pillars"},
        "mu": {"type": "number", "default": POROUS_VISCOSITY_DEFAULT, "description": "Dynamic viscosity"},
        "pressure_drop": {"type": "number", "default": POROUS_PRESSURE_DROP_DEFAULT,
                            "description": "Prescribed pressure drop across the channel (inflow minus outflow)"},
    },
)
def porous_lattice(
    core=None, *, nx=4, ny=4, pillar_radius=0.08, h_pillar=0.015, h_far=0.05,
    mu=POROUS_VISCOSITY_DEFAULT, pressure_drop=POROUS_PRESSURE_DROP_DEFAULT,
):
    return {
        "porous_flow": {
            "_type": "step",
            "address": "local:PorousFlowStep",
            "config": {
                "nx": nx, "ny": ny, "pillar_radius": pillar_radius,
                "h_pillar": h_pillar, "h_far": h_far,
                "mu": mu, "pressure_drop": pressure_drop,
            },
            "inputs": {},
            "outputs": {
                "velocity_x": ["stores", "velocity_x"],
                "velocity_y": ["stores", "velocity_y"],
                "pressure": ["stores", "pressure"],
                "porosity": ["stores", "porosity"],
                "mean_ux": ["stores", "mean_ux"],
                "k_eff": ["stores", "k_eff"],
                "divergence_mean": ["stores", "divergence_mean"],
                "noslip_max_speed": ["stores", "noslip_max_speed"],
                "n_cells": ["stores", "n_cells"],
            },
        },
        "stores": {
            "velocity_x": [],
            "velocity_y": [],
            "pressure": [],
            "porosity": 0.0,
            "mean_ux": 0.0,
            "k_eff": 0.0,
            "divergence_mean": 0.0,
            "noslip_max_speed": 0.0,
            "n_cells": 0,
        },
        "emitter": {
            "_type": "step",
            "address": "local:RAMEmitter",
            "config": {
                "emit": {
                    "velocity_x": "array[float]",
                    "velocity_y": "array[float]",
                    "pressure": "array[float]",
                    "porosity": "float",
                    "mean_ux": "float",
                    "k_eff": "float",
                    "divergence_mean": "float",
                    "noslip_max_speed": "float",
                    "n_cells": "integer",
                },
            },
            "inputs": {
                "velocity_x": ["stores", "velocity_x"],
                "velocity_y": ["stores", "velocity_y"],
                "pressure": ["stores", "pressure"],
                "porosity": ["stores", "porosity"],
                "mean_ux": ["stores", "mean_ux"],
                "k_eff": ["stores", "k_eff"],
                "divergence_mean": ["stores", "divergence_mean"],
                "noslip_max_speed": ["stores", "noslip_max_speed"],
                "n_cells": ["stores", "n_cells"],
            },
        },
    }
