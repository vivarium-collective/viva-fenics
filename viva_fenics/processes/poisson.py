"""PoissonSolverStep: a process-bigraph Step wrapping the FEniCSx MMS Poisson
solve in viva_fenics.fem.

This is a Step (not a Process) because it is stateless and has no
time-varying inputs -- the Port-Design "no inputs => Step" rule. Given only
its config (mesh resolution + polynomial degree), it builds a mesh, solves
the manufactured-solution Poisson problem, and reports the nodal solution
plus its L2 error against the known analytic solution.
"""

import numpy as np
import ufl
from process_bigraph import Step

from viva_fenics import fem

# Manufactured solution for -div(grad(u)) = -6 (source_value default):
#   u_exact(x, y) = 1 + x**2 + 2*y**2
# so that -Laplacian(u_exact) == -6 everywhere, and the Dirichlet BC on the
# unit-square boundary is just u_exact restricted to the boundary.

# Second manufactured solution ("smooth_trig"), used for high-order accuracy
# verification (studies/poisson-validation): u_exact = sin(pi*x)*sin(pi*y),
# a C-infinity function on the unit square with a non-constant Laplacian --
# unlike the quadratic family above, it is NOT exactly representable in any
# fixed-degree Lagrange trial space, so P1/P2/P3 each show a genuine,
# non-trivial discretization error that should converge at its OWN optimal
# L2 rate (degree+1) rather than collapsing to round-off. u_exact is zero on
# the whole unit-square boundary, so the Dirichlet BC is homogeneous.


def _smooth_trig_source(x):
    """f = -Laplacian(u_exact) for u_exact = sin(pi x) sin(pi y)."""
    return 2.0 * np.pi**2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def _smooth_trig_bc(x):
    """u_exact is 0 on the entire unit-square boundary."""
    return 0.0 * x[0]


def _smooth_trig_exact_ufl(x):
    """u_exact as a symbolic UFL expression (for elevated-quadrature L2
    error via `fem.l2_error_exact`, not FE interpolation)."""
    return ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def _make_exact_fn(source_value):
    """Return the exact solution callable for -Laplacian(u) = source_value.

    Only the canonical source_value=-6.0 (see class docstring) has a known
    closed-form exact solution wired here; other values reuse the same
    quadratic family scaled so its Laplacian is `source_value` -- consistent
    with the MMS convention this Step implements.
    """
    # u = 1 + x**2 + 2*y**2 has Laplacian = 2 + 4 = 6, so -Laplacian(u) = -6.
    # For a general source_value, scale the quadratic terms so
    # -Laplacian(u) == source_value while keeping the same family/shape.
    scale = source_value / -6.0

    def exact_fn(x):
        return 1 + scale * (x[0] ** 2 + 2 * x[1] ** 2)

    return exact_fn


class PoissonSolverStep(Step):
    """Solve the steady Poisson equation on a unit-square mesh and report
    the nodal solution plus its L2 error against a manufactured exact
    solution.

    Two manufactured-solution families are supported via `problem`:

    - "quadratic" (default): u_exact = 1 + x^2 + 2*y^2, constant Laplacian,
      exactly representable for degree>=2 (Galerkin orthogonality -> error
      at round-off) -- the original correctness MMS.
    - "smooth_trig": u_exact = sin(pi*x)*sin(pi*y), a smooth (C-infinity)
      but non-polynomial solution -- no fixed-degree Lagrange space
      represents it exactly, so P1/P2/P3 each show a genuine L2
      discretization error converging at its own optimal rate (degree+1).
      Used by the high-order accuracy verification (studies/
      poisson-validation): the L2 error is computed via
      `fem.l2_error_exact` (elevated-quadrature comparison against the
      symbolic UFL exact expression) rather than `fem.l2_error`
      (FE-interpolated exact solution), since interpolating the exact
      solution into the same degree-p space first would understate exactly
      the higher-order error a convergence-rate check needs to see.
    """

    config_schema = {
        "resolution": {"_type": "integer", "_default": 16},
        "degree": {"_type": "integer", "_default": 2},
        "source_value": {"_type": "float", "_default": -6.0},
        "problem": {"_type": "string", "_default": "quadratic"},
    }

    def inputs(self):
        return {}

    def outputs(self):
        return {
            "solution": "array[float]",
            "l2_error": "float",
        }

    def update(self, state):
        resolution = self.config["resolution"]
        degree = self.config["degree"]
        problem = self.config.get("problem", "quadratic")

        domain, V = fem.build_mesh("unit_square", resolution, degree=degree)

        if problem == "smooth_trig":
            uh = fem.solve_poisson(
                domain, V,
                source_fn=_smooth_trig_source,
                bc_fn=_smooth_trig_bc,
            )
            error = fem.l2_error_exact(domain, V, uh, _smooth_trig_exact_ufl)
        elif problem == "quadratic":
            source_value = self.config["source_value"]
            exact_fn = _make_exact_fn(source_value)
            uh = fem.solve_poisson(
                domain, V,
                source_fn=lambda x: source_value + 0 * x[0],
                bc_fn=exact_fn,
            )
            error = fem.l2_error(domain, V, uh, exact_fn)
        else:
            raise ValueError(f"Unsupported problem: {problem!r}")

        return {
            "solution": uh,
            "l2_error": error,
        }
