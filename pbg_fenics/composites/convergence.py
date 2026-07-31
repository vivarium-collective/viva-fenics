"""Composite generator for a single-resolution Poisson solve used as the
mesh-convergence readout.

Wires a single ``PoissonSolverStep`` (Task 2) to a RAM emitter, exactly like
``poisson_baseline`` (Task 3's composite), so the dashboard's Composites tab
/ a bare workspace run can build and execute it at one (resolution, degree)
pair. The convergence *sweep* itself is NOT expressed here -- a real
convergence study is a Study with variants over ``resolution`` (Task 7);
this generator only parameterizes a single point on that sweep, reporting
``l2_error`` as the per-point convergence readout.
"""

from __future__ import annotations

from viva_superpowers.composite_generator import composite_generator


@composite_generator(
    name="mesh_convergence",
    description=(
        "Poisson L2 error vs mesh resolution -- FEM convergence "
        "(error ∝ h^(degree+1))."
    ),
    parameters={
        "resolution": {"type": "integer", "default": 16,
                        "description": "Mesh cells per side of the unit square"},
        "degree": {"type": "integer", "default": 1,
                    "description": "Lagrange element polynomial degree"},
    },
)
def mesh_convergence(core=None, *, resolution=16, degree=1):
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
