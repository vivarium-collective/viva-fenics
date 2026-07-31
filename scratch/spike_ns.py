"""Spike: validate dolfinx 0.11 API for lid-driven-cavity IPCS before
writing viva_fenics/processes/flow.py.
"""
import itertools

import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc

_prefix = itertools.count()

RESOLUTION = 16
RE = 100.0
U_LID = 1.0
RHO = 1.0
NU = U_LID * 1.0 / RE  # L=1
MU = RHO * NU
DT = 0.005
NSTEPS = 200

domain = mesh.create_unit_square(
    MPI.COMM_WORLD, RESOLUTION, RESOLUTION, diagonal=mesh.DiagonalType.crossed
)
gdim = domain.geometry.dim
V = fem.functionspace(domain, ("Lagrange", 2, (gdim,)))
Q = fem.functionspace(domain, ("Lagrange", 1))

print("V dofmap num dofs (blocked/nodes):", V.dofmap.index_map.size_local, "bs", V.dofmap.index_map_bs)
coords_v = V.tabulate_dof_coordinates()
print("V.tabulate_dof_coordinates shape:", coords_v.shape)
coords_q = Q.tabulate_dof_coordinates()
print("Q.tabulate_dof_coordinates shape:", coords_q.shape)

u_n = fem.Function(V)
print("u_n.x.array shape:", u_n.x.array.shape)

# boundary conditions
fdim = domain.topology.dim - 1


def lid(x):
    return np.isclose(x[1], 1.0)


def walls(x):
    return np.logical_or(
        np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)),
        np.isclose(x[1], 0.0),
    )


lid_facets = mesh.locate_entities_boundary(domain, fdim, lid)
wall_facets = mesh.locate_entities_boundary(domain, fdim, walls)
lid_dofs = fem.locate_dofs_topological(V, fdim, lid_facets)
wall_dofs = fem.locate_dofs_topological(V, fdim, wall_facets)
print("lid_dofs", len(lid_dofs), "wall_dofs", len(wall_dofs))

bc_lid = fem.dirichletbc(PETSc.ScalarType((U_LID, 0.0)), lid_dofs, V)
bc_wall = fem.dirichletbc(PETSc.ScalarType((0.0, 0.0)), wall_dofs, V)
bcu = [bc_lid, bc_wall]


def corner(x):
    return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))


p_dofs = fem.locate_dofs_geometrical(Q, corner)
print("p_dofs", p_dofs)
bc_p = fem.dirichletbc(PETSc.ScalarType(0.0), p_dofs, Q)
bcp = [bc_p]

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
p = ufl.TrialFunction(Q)
q = ufl.TestFunction(Q)

u_ = fem.Function(V)  # tentative velocity
p_n = fem.Function(Q)
p_ = fem.Function(Q)

k = fem.Constant(domain, PETSc.ScalarType(DT))
mu = fem.Constant(domain, PETSc.ScalarType(MU))
rho = fem.Constant(domain, PETSc.ScalarType(RHO))


def epsilon(uu):
    return ufl.sym(ufl.nabla_grad(uu))


def sigma(uu, pp):
    return 2 * mu * epsilon(uu) - pp * ufl.Identity(gdim)


U_avg = 0.5 * (u_n + u)

F1 = rho / k * ufl.dot(u - u_n, v) * ufl.dx
F1 += rho * ufl.dot(ufl.dot(u_n, ufl.nabla_grad(u_n)), v) * ufl.dx
F1 += ufl.inner(sigma(U_avg, p_n), epsilon(v)) * ufl.dx
a1 = ufl.lhs(F1)
L1 = ufl.rhs(F1)

a2 = ufl.dot(ufl.grad(p), ufl.grad(q)) * ufl.dx
L2 = ufl.dot(ufl.grad(p_n), ufl.grad(q)) * ufl.dx - (rho / k) * ufl.div(u_) * q * ufl.dx

a3 = ufl.dot(u, v) * ufl.dx
L3 = ufl.dot(u_, v) * ufl.dx - k * ufl.dot(ufl.grad(p_ - p_n), v) * ufl.dx

import time

t0 = time.time()
for step in range(NSTEPS):
    prefix1 = f"spike_ns_1_{next(_prefix)}_"
    problem1 = LinearProblem(
        a1, L1, bcs=bcu, u=u_,
        petsc_options_prefix=prefix1,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    problem1.solve()

    prefix2 = f"spike_ns_2_{next(_prefix)}_"
    problem2 = LinearProblem(
        a2, L2, bcs=bcp, u=p_,
        petsc_options_prefix=prefix2,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    problem2.solve()

    prefix3 = f"spike_ns_3_{next(_prefix)}_"
    problem3 = LinearProblem(
        a3, L3, bcs=[], u=u_n,
        petsc_options_prefix=prefix3,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    problem3.solve()
    # Note: solving into u_n directly via `u=u_n` for step 3 updates u_n
    # in place for the next iteration's convection term.
    p_n.x.array[:] = p_.x.array

    if step % 40 == 0:
        speed = np.sqrt(u_n.x.array[0::2] ** 2 + u_n.x.array[1::2] ** 2)
        print(step, "max speed", speed.max(), "elapsed", time.time() - t0)

speed = np.sqrt(u_n.x.array[0::2] ** 2 + u_n.x.array[1::2] ** 2)
print("final max speed", speed.max(), "total time", time.time() - t0)
print("u_n.x.array len", len(u_n.x.array), "coords_v rows", coords_v.shape[0])

div_form = fem.form(ufl.div(u_n) * ufl.div(u_n) * ufl.dx)
div_l2 = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(div_form), op=MPI.SUM))
print("div L2 norm (integral form)", div_l2)

# Interpolate div(u_n) onto a DG0 space to get a per-cell scalar reading
# (more interpretable than the raw L2 integral, and lets us exclude the
# known lid-corner singularities where the moving-lid/no-slip BC is
# discontinuous).
DG0 = fem.functionspace(domain, ("DG", 0))
div_expr = fem.Expression(ufl.div(u_n), DG0.element.interpolation_points)
div_field = fem.Function(DG0)
div_field.interpolate(div_expr)
cell_coords = DG0.tabulate_dof_coordinates()[:, :2]
dist_to_top_corners = np.minimum(
    np.hypot(cell_coords[:, 0] - 0.0, cell_coords[:, 1] - 1.0),
    np.hypot(cell_coords[:, 0] - 1.0, cell_coords[:, 1] - 1.0),
)
mask_interior = dist_to_top_corners > 0.1
print("div DG0: mean|div|", np.mean(np.abs(div_field.x.array)), "max|div|", np.max(np.abs(div_field.x.array)))
print(
    "div DG0 (excluding lid corners r>0.1): mean|div|",
    np.mean(np.abs(div_field.x.array[mask_interior])),
    "max|div|",
    np.max(np.abs(div_field.x.array[mask_interior])),
)
