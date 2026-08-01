"""Composite generator for a standalone complex-geometry (gmsh) Poisson
solve.

Wires a single ``ComplexGeometryStep`` (real gmsh geometry -> dolfinx import
-> Poisson solve, see ``viva_fenics.fem_gmsh`` / ``viva_fenics.processes.
complex_geometry``) to a RAM emitter so the dashboard's Composites tab / a
bare workspace run can build and execute it. The step is stateless (no
inputs) -- it builds the gmsh geometry and solves fresh from config on every
invocation -- so the composite's ``stores`` block only needs placeholder
slots for the step's outputs to land in, same convention as
``poisson_baseline``.
"""

from __future__ import annotations

from viva_superpowers.composite_generator import composite_generator


@composite_generator(
    name="complex_geometry",
    description="Poisson/diffusion on gmsh-generated non-trivial domains (obstacle/L-shape/annulus).",
    parameters={
        "geometry": {"type": "string", "default": "obstacle",
                      "description": "gmsh domain: 'obstacle' (square w/ circular hole), 'lshape', or 'annulus'"},
        "resolution": {"type": "integer", "default": 32,
                        "description": "Target gmsh mesh element-size divisor (h = 1/resolution)"},
    },
)
def complex_geometry(core=None, *, geometry="obstacle", resolution=32):
    return {
        "complex_geometry": {
            "_type": "step",
            "address": "local:ComplexGeometryStep",
            "config": {"geometry": geometry, "resolution": resolution},
            "inputs": {},
            "outputs": {
                "solution": ["stores", "solution"],
                "n_cells": ["stores", "n_cells"],
                "solution_max": ["stores", "solution_max"],
            },
        },
        "stores": {
            "solution": [],
            "n_cells": 0,
            "solution_max": 0.0,
        },
        "emitter": {
            "_type": "step",
            "address": "local:RAMEmitter",
            "config": {
                "emit": {
                    "solution": "array[float]",
                    "n_cells": "integer",
                    "solution_max": "float",
                },
            },
            "inputs": {
                "solution": ["stores", "solution"],
                "n_cells": ["stores", "n_cells"],
                "solution_max": ["stores", "solution_max"],
            },
        },
    }
