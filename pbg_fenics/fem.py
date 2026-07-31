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
    domain = mesh.create_unit_square(MPI.COMM_WORLD, resolution, resolution)
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

    prefix = f"pbg_fenics_poisson_{next(_prefix_counter)}_"
    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix=prefix,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    uh = problem.solve()
    return uh.x.array.copy()


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
