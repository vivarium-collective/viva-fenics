"""PoissonSolverStep: a process-bigraph Step wrapping the FEniCSx MMS Poisson
solve in pbg_fenics.fem.

This is a Step (not a Process) because it is stateless and has no
time-varying inputs -- the Port-Design "no inputs => Step" rule. Given only
its config (mesh resolution + polynomial degree), it builds a mesh, solves
the manufactured-solution Poisson problem, and reports the nodal solution
plus its L2 error against the known analytic solution.
"""

from process_bigraph import Step

from pbg_fenics import fem

# Manufactured solution for -div(grad(u)) = -6 (source_value default):
#   u_exact(x, y) = 1 + x**2 + 2*y**2
# so that -Laplacian(u_exact) == -6 everywhere, and the Dirichlet BC on the
# unit-square boundary is just u_exact restricted to the boundary.


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
    the nodal solution plus its L2 error against the manufactured exact
    solution.
    """

    config_schema = {
        "resolution": {"_type": "integer", "_default": 16},
        "degree": {"_type": "integer", "_default": 2},
        "source_value": {"_type": "float", "_default": -6.0},
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
        source_value = self.config["source_value"]

        exact_fn = _make_exact_fn(source_value)

        domain, V = fem.build_mesh("unit_square", resolution, degree=degree)
        uh = fem.solve_poisson(
            domain, V,
            source_fn=lambda x: source_value + 0 * x[0],
            bc_fn=exact_fn,
        )
        error = fem.l2_error(domain, V, uh, exact_fn)

        return {
            "solution": uh,
            "l2_error": error,
        }
