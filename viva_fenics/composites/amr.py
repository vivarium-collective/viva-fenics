"""Composite generator for the AMR study's baseline composite: a single
``AdaptiveRefinementStep`` wired to a RAM emitter, exactly like
``mesh_convergence``/``poisson_baseline``, so the dashboard's Composites tab
/ a bare workspace run can build and execute the AMR loop at one
(n_refinements, marking_fraction, degree) parameterization.

The full uniform-vs-adaptive COMPARISON (the study's actual point) lives in
``studies/mesh-convergence/sims/run.py``, which calls
``viva_fenics.fem_amr.run_amr_loop``/``run_uniform_loop`` directly for the
richer sweep + mesh-geometry snapshots the comparison chart/animation need;
this generator only needs to expose a real, buildable composite id for the
study.
"""

from __future__ import annotations

from viva_superpowers.composite_generator import composite_generator


@composite_generator(
    name="adaptive_refinement",
    description=(
        "Residual-based adaptive mesh refinement (AMR) on the L-shaped "
        "domain's re-entrant-corner Laplace singularity -- Doerfler "
        "marking concentrates elements at the corner, recovering "
        "near-optimal O(dofs^-1/2) energy-norm convergence vs the "
        "O(dofs^-1/3) rate uniform refinement is capped at by the "
        "r^(2/3) corner singularity."
    ),
    parameters={
        "n_refinements": {"type": "integer", "default": 8,
                            "description": "Number of Doerfler-mark-and-refine levels"},
        "marking_fraction": {"type": "number", "default": 0.5,
                               "description": "Doerfler marking fraction theta (0,1]"},
        "degree": {"type": "integer", "default": 1,
                    "description": "Lagrange element polynomial degree"},
    },
)
def adaptive_refinement(core=None, *, n_refinements=8, marking_fraction=0.5, degree=1):
    return {
        "amr": {
            "_type": "step",
            "address": "local:AdaptiveRefinementStep",
            "config": {
                "n_refinements": n_refinements,
                "marking_fraction": marking_fraction,
                "degree": degree,
            },
            "inputs": {},
            "outputs": {
                "solution": ["stores", "solution"],
                "energy_error": ["stores", "energy_error"],
                "dofs_history": ["stores", "dofs_history"],
                "error_history": ["stores", "error_history"],
            },
        },
        "stores": {
            "solution": [],
            "energy_error": 0.0,
            "dofs_history": [],
            "error_history": [],
        },
        "emitter": {
            "_type": "step",
            "address": "local:RAMEmitter",
            "config": {
                "emit": {
                    "solution": "array[float]",
                    "energy_error": "float",
                    "dofs_history": "array[float]",
                    "error_history": "array[float]",
                },
            },
            "inputs": {
                "solution": ["stores", "solution"],
                "energy_error": ["stores", "energy_error"],
                "dofs_history": ["stores", "dofs_history"],
                "error_history": ["stores", "error_history"],
            },
        },
    }
