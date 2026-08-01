"""ComplexGeometryStep: a process-bigraph Step wrapping a REAL Poisson solve
on a gmsh-generated non-trivial 2D domain (obstacle / L-shape / annulus --
see ``viva_fenics.fem_gmsh``'s module docstring).

Like ``PoissonSolverStep`` (Task 2), this is a Step (not a Process): it is
stateless and has no time-varying inputs -- the Port-Design "no inputs =>
Step" rule. Given only its config (named geometry + mesh resolution +
element degree), it builds the geometry with gmsh, imports it into dolfinx,
solves -Laplacian(u) = source_value with a homogeneous Dirichlet boundary
condition, and reports the nodal solution plus two scalar readouts: the
imported mesh's cell count (confirms the gmsh->dolfinx import actually
produced a non-trivial triangulation) and the solution's max value (confirms
a non-trivial, non-zero solution was actually computed).
"""
from __future__ import annotations

import numpy as np
from process_bigraph import Step

from viva_fenics import fem_gmsh


class ComplexGeometryStep(Step):
    """Solve the Poisson equation on a gmsh-generated non-trivial domain and
    report the nodal solution plus the imported mesh's cell count and the
    solution's max value.
    """

    config_schema = {
        "geometry": {"_type": "string", "_default": "obstacle"},
        "resolution": {"_type": "integer", "_default": 32},
        "degree": {"_type": "integer", "_default": 1},
        "source_value": {"_type": "float", "_default": 1.0},
    }

    def inputs(self):
        return {}

    def outputs(self):
        return {
            "solution": "array[float]",
            "n_cells": "integer",
            "solution_max": "float",
        }

    def update(self, state):
        geometry = self.config["geometry"]
        resolution = self.config["resolution"]
        degree = self.config["degree"]
        source_value = self.config["source_value"]

        domain, V = fem_gmsh.build_gmsh_mesh(geometry, resolution, degree=degree)
        uh = fem_gmsh.solve_poisson_gmsh(domain, V, source_value=source_value)

        return {
            "solution": uh,
            "n_cells": fem_gmsh.n_cells(domain),
            "solution_max": float(np.max(uh)),
        }
