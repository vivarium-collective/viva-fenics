"""Spike: dolfinx 0.11's adaptive-refinement API, verified end-to-end on the
L-shaped re-entrant-corner Laplace singularity, before writing the real
``viva_fenics.fem_amr`` module (that module's docstrings explain the
math/estimator in full; this file is the exploratory scratch-pad that
established the API calls actually work).

Findings (informed the real implementation):

- ``fem.Expression(u_exact_ufl, V.element.interpolation_points)`` (an
  ATTRIBUTE, not a method -- ``interpolation_points()`` raises
  "'numpy.ndarray' object is not callable" in this dolfinx version) lets
  ``Function.interpolate(...)`` consume a real symbolic UFL expression
  (built with ``ufl.atan2``/``ufl.sqrt``) directly -- no need to hand-roll a
  numpy callable duplicating UFL's branch-cut convention.
- ``ufl.FacetArea(domain)`` must be restricted (``he("+")``) inside a ``dS``
  interior-facet form, even though it is numerically the same from both
  sides -- UFL raises "Discontinuous type CellFacetJacobian must be
  restricted" otherwise.
- Assembling a linear form ``he("+") * ufl.avg(w) * jump(grad(uh), n)**2 *
  dS`` with ``w`` a DG0 TestFunction directly produces the standard
  residual-estimator's per-cell eta_T^2 array (DG0 dof numbering == cell
  numbering) -- ``avg(w)`` splits each interior edge's contribution evenly
  between its two adjacent cells' dofs.
- ``dolfinx.mesh.refine(domain, edges=marked_edges)`` (edges from
  ``mesh.compute_incident_entities(topology, marked_cells, tdim, 1)``) does
  real Plaza/NVB conforming refinement -- confirmed the corner-adjacent
  cells get marked (Dorfler theta=0.5 picked only 4 of 190 cells at this
  mesh, all near the corner) and the new mesh remains conforming.
- The residual indicators are dramatically larger near the corner than far
  away (measured ~150x mean ratio at this mesh) -- direct confirmation the
  estimator is doing its job before building the full AMR loop.

Real dolfinx throughout. Run: ``pixi run python scratch/spike_amr.py``
"""
import time

import gmsh
import numpy as np
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmsh as dgmsh
from mpi4py import MPI


def build_lshape(h):
    """Omega = [-1,1]^2 \\ [0,1]x[-1,0]: re-entrant corner at the ORIGIN
    (distinct from fem_gmsh's "lshape", whose corner sits at (0.5,0.5))."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("lshape_amr")
        big = gmsh.model.occ.addRectangle(-1, -1, 0, 2, 2)
        notch = gmsh.model.occ.addRectangle(0, -1, 0, 1, 1)
        cut, _ = gmsh.model.occ.cut([(2, big)], [(2, notch)])
        gmsh.model.occ.synchronize()
        surf_tag = cut[0][1]
        gmsh.model.addPhysicalGroup(2, [surf_tag], tag=1)
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
        gmsh.model.mesh.generate(2)
        mesh_data = dgmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
        domain = mesh_data.mesh
    finally:
        gmsh.finalize()
    return domain


def exact_ufl(x):
    """u(r,theta) = r^(2/3) sin(2 theta/3) -- exact singular harmonic
    solution, theta wrapped into [0, 2pi) so it's single-valued."""
    r = ufl.sqrt(x[0] ** 2 + x[1] ** 2)
    theta = ufl.atan2(x[1], x[0])
    theta = ufl.conditional(ufl.lt(theta, 0), theta + 2 * ufl.pi, theta)
    return r ** (2.0 / 3.0) * ufl.sin(2.0 / 3.0 * theta)


def solve_and_estimate(domain, degree=1):
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)
    u_exact = exact_ufl(x)

    uD = fem.Function(V)
    expr = fem.Expression(u_exact, V.element.interpolation_points)
    uD.interpolate(expr)

    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, tdim)
    bfacets = dmesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, tdim - 1, bfacets)
    bc = fem.dirichletbc(uD, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = fem.Constant(domain, 0.0) * v * ufl.dx  # f=0 (Laplace)

    problem = LinearProblem(
        a, L, bcs=[bc], petsc_options_prefix=f"amr_spike_{id(domain)}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    uh = problem.solve()

    dx_md = ufl.dx(metadata={"quadrature_degree": 12})
    diff = ufl.grad(uh) - ufl.grad(u_exact)
    err_form = fem.form(ufl.dot(diff, diff) * dx_md)
    err_local = fem.assemble_scalar(err_form)
    energy_err = float(np.sqrt(domain.comm.allreduce(err_local, op=MPI.SUM)))

    DG0 = fem.functionspace(domain, ("DG", 0))
    w = ufl.TestFunction(DG0)
    n = ufl.FacetNormal(domain)
    he = ufl.FacetArea(domain)
    jump_sq = ufl.jump(ufl.grad(uh), n) ** 2
    eta_form = fem.form(he("+") * ufl.avg(w) * jump_sq * ufl.dS)
    eta_vec = fem.assemble_vector(eta_form)
    eta_sq = eta_vec.array.copy()
    ndofs = V.dofmap.index_map.size_global
    return uh, V, energy_err, eta_sq, ndofs


if __name__ == "__main__":
    domain = build_lshape(0.2)
    tdim = domain.topology.dim
    print("initial cells", domain.topology.index_map(tdim).size_local)

    uh, V, err, eta_sq, ndofs = solve_and_estimate(domain)
    print("dofs", ndofs, "energy err", err, "total eta", np.sqrt(eta_sq.sum()), "n indicators", len(eta_sq))

    midpoints = dmesh.compute_midpoints(
        domain, tdim, np.arange(domain.topology.index_map(tdim).size_local, dtype=np.int32)
    )
    dist = np.hypot(midpoints[:, 0], midpoints[:, 1])
    near = dist < 0.15
    far = dist > 0.5
    print("mean eta_sq near corner:", eta_sq[near].mean() if near.any() else None)
    print("mean eta_sq far:", eta_sq[far].mean() if far.any() else None)

    theta = 0.5
    order = np.argsort(eta_sq)[::-1]
    cum = np.cumsum(eta_sq[order])
    total = eta_sq.sum()
    cutoff = np.searchsorted(cum, theta * total) + 1
    marked_cells = order[:cutoff].astype(np.int32)
    print("marked", len(marked_cells), "of", len(eta_sq))

    domain.topology.create_connectivity(tdim, 1)
    domain.topology.create_connectivity(1, tdim)
    marked_edges = dmesh.compute_incident_entities(domain.topology, marked_cells, tdim, 1)
    marked_edges = np.unique(marked_edges).astype(np.int32)
    print("marked edges", len(marked_edges))

    t0 = time.time()
    new_domain, parent_cells, parent_facets = dmesh.refine(domain, edges=marked_edges)
    print("refine time", time.time() - t0)
    print("new cells", new_domain.topology.index_map(new_domain.topology.dim).size_local)
