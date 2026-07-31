"""Composite generator for a standalone Poisson solve.

Wires a single ``PoissonSolverStep`` (Task 2) to a RAM emitter so the
dashboard's Composites tab / a bare workspace run can build and execute it.
The step is stateless (no inputs) -- it computes its solution + L2 error
straight from config on invocation -- so the composite's ``stores`` block
only needs placeholder slots for the step's outputs to land in.
"""

from __future__ import annotations

from viva_superpowers.composite_generator import composite_generator


@composite_generator(
    name="poisson_baseline",
    description="Steady Poisson solve validated vs analytic (MMS).",
    parameters={
        "resolution": {"type": "integer", "default": 16,
                        "description": "Mesh cells per side of the unit square"},
        "degree": {"type": "integer", "default": 2,
                    "description": "Lagrange element polynomial degree"},
    },
)
def poisson_baseline(core=None, *, resolution=16, degree=2):
    return {
        "poisson": {
            "_type": "step",
            "address": "local:PoissonSolverStep",
            "config": {"resolution": resolution, "degree": degree},
            "inputs": {},
            "outputs": {
                "solution": ["stores", "solution"],
                "l2_error": ["stores", "l2_error"],
            },
        },
        "stores": {
            "solution": [],
            "l2_error": 0.0,
        },
        "emitter": {
            "_type": "step",
            "address": "local:RAMEmitter",
            "config": {"emit": {"solution": "array[float]", "l2_error": "float"}},
            "inputs": {
                "solution": ["stores", "solution"],
                "l2_error": ["stores", "l2_error"],
            },
        },
    }
