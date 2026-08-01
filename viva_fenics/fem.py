"""dolfinx helper functions for building meshes, solving Poisson problems,
and extracting nodal data for process-bigraph wrapping.

Validated against dolfinx 0.11.0 (see ../scratch/spike_poisson.py).
"""

import itertools

import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

# LinearProblem requires a unique-ish petsc_options_prefix per call in
# dolfinx 0.11; use a counter so repeated solves in one process don't clash.
_prefix_counter = itertools.count()


def build_mesh(kind, resolution, degree=1):
    """Build a mesh + function space.

    Args:
        kind: mesh kind, currently only "unit_square" is supported.
        resolution: number of cells per side.
        degree: Lagrange element degree.

    Returns:
        (domain, V) tuple.
    """
    if kind != "unit_square":
        raise ValueError(f"Unsupported mesh kind: {kind!r}")
    # dolfinx's default unit-square triangulation (diagonal="right", every
    # quad cut the same way) makes P1 (degree=1) Lagrange elements *nodally
    # exact* for any manufactured solution with constant Laplacian -- i.e.
    # any quadratic, including this module's MMS family -- a well-known
    # structured-mesh superconvergence artifact, NOT genuine O(h^2)
    # discretization behavior. That collapses PoissonSolverStep's l2_error to
    # floating-point round-off (~1e-15) at every resolution for degree=1,
    # masking real convergence rate. "crossed" (4 triangles per quad, no
    # uniform diagonal direction) breaks that artifact -- confirmed
    # empirically: degree=1 error now scales as ~h^2 (rate ~2.0 across
    # resolution doublings) while degree>=2 stays exact to ~1e-13 (unaffected,
    # since a quadratic is exactly representable in the P2+ trial space
    # regardless of triangulation -- Galerkin orthogonality, not a mesh
    # property).
    domain = mesh.create_unit_square(
        MPI.COMM_WORLD, resolution, resolution,
        diagonal=mesh.DiagonalType.crossed,
    )
    V = fem.functionspace(domain, ("Lagrange", degree))
    return domain, V


def solve_poisson(domain, V, source_fn, bc_fn):
    """Solve the Poisson equation -div(grad(u)) = f with Dirichlet BC.

    Args:
        domain: dolfinx mesh.
        V: dolfinx function space over domain.
        source_fn: callable x (shape (gdim, npoints)) -> f values at x.
        bc_fn: callable x (shape (gdim, npoints)) -> Dirichlet values at x.

    Returns:
        np.ndarray of nodal solution values (uh.x.array copy).
    """
    uD = fem.Function(V)
    uD.interpolate(bc_fn)

    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, tdim - 1, boundary_facets)
    bc = fem.dirichletbc(uD, dofs)

    f = fem.Function(V)
    f.interpolate(source_fn)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    prefix = f"viva_fenics_poisson_{next(_prefix_counter)}_"
    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix=prefix,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    uh = problem.solve()
    return uh.x.array.copy()


def diffusion_step(domain, V, u_n_array, source_array, dt, D, bc_value=None):
    """Advance one backward-Euler step of the transient diffusion equation.

    Solves for u given the previous nodal field u_n and an additive nodal
    source term:

        (u*v + dt*D*dot(grad(u), grad(v))) dx = (u_n + dt*source) * v dx

    By default no Dirichlet boundary condition is applied, so the boundary is
    natural Neumann (zero-flux) -- mass is conserved up to the source term.

    Args:
        domain: dolfinx mesh.
        V: dolfinx function space over domain.
        u_n_array: nodal values of the field at the start of the step.
        source_array: nodal values of the additive source over the step.
        dt: step size.
        D: diffusion coefficient.
        bc_value: if given, pins the field to this constant value on the
            ``x=0`` face of the domain via a real Dirichlet BC (e.g. a
            morphogen production boundary c=c0), leaving every other face
            natural Neumann (zero-flux). ``None`` (the default) preserves
            the original all-Neumann behavior exactly.

    Returns:
        np.ndarray of nodal solution values after the step (uh.x.array copy).
    """
    u_n = fem.Function(V)
    u_n.x.array[:] = u_n_array

    source = fem.Function(V)
    source.x.array[:] = source_array

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = (u * v + dt * D * ufl.dot(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n + dt * source) * v * ufl.dx

    bcs = []
    if bc_value is not None:
        tdim = domain.topology.dim
        domain.topology.create_connectivity(tdim - 1, tdim)
        boundary_facets = mesh.locate_entities_boundary(
            domain, tdim - 1, lambda x: np.isclose(x[0], 0.0)
        )
        dofs = fem.locate_dofs_topological(V, tdim - 1, boundary_facets)
        bc_fn = fem.Function(V)
        bc_fn.x.array[:] = bc_value
        bcs = [fem.dirichletbc(bc_fn, dofs)]

    prefix = f"viva_fenics_diff_{next(_prefix_counter)}_"
    problem = LinearProblem(
        a, L, bcs=bcs,
        petsc_options_prefix=prefix,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    uh = problem.solve()
    return uh.x.array.copy()


def field_integral(domain, V, array):
    """Integrate a nodal field over the domain: returns a float scalar of
    integral(u) dx, computed via dolfinx's assemble_scalar (consistent with
    the FEM mass matrix, not a naive nodal sum).
    """
    uh = fem.Function(V)
    uh.x.array[:] = array
    form = fem.form(uh * ufl.dx)
    local = fem.assemble_scalar(form)
    return float(domain.comm.allreduce(local, op=MPI.SUM))


def l2_error_exact(domain, V, uh_array, exact_ufl_fn, quadrature_degree=None):
    """Compute the L2 error between a nodal solution and a SYMBOLIC exact
    expression, at elevated quadrature -- for high-order accuracy
    verification, where interpolating the exact solution into the same
    degree-p trial space first (as `l2_error` does) would understate the
    true error: it measures ||uh - I_p(u)|| (distance between two
    degree-p objects) rather than ||uh - u||, silently discarding exactly
    the higher-order information a convergence-rate check needs. Here `u`
    is built as a real UFL expression via `ufl.SpatialCoordinate(domain)`
    (e.g. `ufl.sin`), so the error form can be integrated at a quadrature
    degree well above V's polynomial degree instead.

    Args:
        domain: dolfinx mesh.
        V: dolfinx function space over domain.
        uh_array: nodal values, e.g. returned by solve_poisson.
        exact_ufl_fn: callable(x) -> UFL expression, where x is
            `ufl.SpatialCoordinate(domain)` (NOT a numpy callable like
            `l2_error`'s `exact_fn` -- this one builds symbolic UFL, e.g.
            ``lambda x: ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])``).
        quadrature_degree: quadrature metadata degree; defaults to
            ``2 * (degree + 3)``, comfortably resolving a smooth
            trigonometric manufactured solution without becoming the
            accuracy bottleneck at any of P1/P2/P3.

    Returns:
        L2 norm of (uh - u_exact) as a float.
    """
    uh = fem.Function(V)
    uh.x.array[:] = uh_array

    x = ufl.SpatialCoordinate(domain)
    u_exact = exact_ufl_fn(x)

    degree = V.element.basix_element.degree
    if quadrature_degree is None:
        quadrature_degree = 2 * (degree + 3)
    dx_q = ufl.dx(metadata={"quadrature_degree": quadrature_degree})

    error_form = fem.form((uh - u_exact) ** 2 * dx_q)
    error_local = fem.assemble_scalar(error_form)
    return float(np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM)))


def node_coords(V):
    """Return dof coordinates for V, shape (N, 2)."""
    return V.tabulate_dof_coordinates()[:, :2]


def l2_error(domain, V, uh_array, exact_fn):
    """Compute the L2 error between a nodal solution array and an exact fn.

    Args:
        domain: dolfinx mesh.
        V: dolfinx function space over domain.
        uh_array: nodal values, e.g. returned by solve_poisson.
        exact_fn: callable x (shape (gdim, npoints)) -> exact values at x.

    Returns:
        L2 norm of (uh - exact) as a float.
    """
    uh = fem.Function(V)
    uh.x.array[:] = uh_array

    u_exact = fem.Function(V)
    u_exact.interpolate(exact_fn)

    error_form = fem.form((uh - u_exact) ** 2 * ufl.dx)
    error_local = fem.assemble_scalar(error_form)
    return float(np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM)))
