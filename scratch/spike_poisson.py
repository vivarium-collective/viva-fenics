import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
import ufl

domain = mesh.create_unit_square(MPI.COMM_WORLD, 16, 16)
V = fem.functionspace(domain, ("Lagrange", 2))

def u_exact_expr(x):
    return 1 + x[0]**2 + 2*x[1]**2

uD = fem.Function(V); uD.interpolate(u_exact_expr)
tdim = domain.topology.dim
domain.topology.create_connectivity(tdim-1, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
dofs = fem.locate_dofs_topological(V, tdim-1, boundary_facets)
bc = fem.dirichletbc(uD, dofs)

u = ufl.TrialFunction(V); v = ufl.TestFunction(V)
f = fem.Constant(domain, -6.0)
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx
problem = LinearProblem(a, L, bcs=[bc],
    petsc_options_prefix="spike_poisson_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# L2 error
error = fem.form((uh - uD)**2 * ufl.dx)
err = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(error), op=MPI.SUM))
print("L2 error:", err)
print("ndofs:", uh.x.array.size)
