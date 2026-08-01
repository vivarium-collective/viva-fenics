"""Spike: validate gmsh Python API -> dolfinx.io.gmsh.model_to_mesh import
for three non-trivial 2D domains, before writing viva_fenics/fem_gmsh.py.

dolfinx 0.11 note: the gmsh-import helpers live at ``dolfinx.io.gmsh``
(module name ``gmshio`` from older dolfinx docs/tutorials was renamed) --
confirmed via ``dolfinx.io.gmsh.model_to_mesh`` inspection, not from stale
memory. ``model_to_mesh`` returns a ``MeshData`` NamedTuple (``.mesh`` is
the dolfinx ``Mesh``), not a bare ``(mesh, cell_tags, facet_tags)`` tuple as
in older dolfinx.

Geometries:
  1. obstacle: unit square [0,1]x[0,1] with a circular hole (r=0.2,
     centered at (0.5, 0.5)) cut out via boolean fragment/cut.
  2. lshape: L-shaped domain (unit square minus its upper-right quadrant).
  3. annulus: annulus between r=0.2 and r=0.5 centered at origin.

For each: build with the gmsh Python API, generate a 2D mesh, import via
model_to_mesh(gdim=2), report #cells, then solve a Poisson problem with a
Dirichlet BC on the full boundary and check the solution is finite.

Run: pixi run python scratch/spike_gmsh.py
"""
from __future__ import annotations

import itertools

import gmsh
import numpy as np
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmsh as dgmsh
from mpi4py import MPI

_prefix = itertools.count()


def build_obstacle(resolution):
    gmsh.initialize()
    gmsh.model.add("obstacle")
    h = 1.0 / resolution
    rect = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
    hole = gmsh.model.occ.addDisk(0.5, 0.5, 0, 0.2, 0.2)
    cut, _ = gmsh.model.occ.cut([(2, rect)], [(2, hole)])
    gmsh.model.occ.synchronize()
    surf_tag = cut[0][1]
    gmsh.model.addPhysicalGroup(2, [surf_tag], tag=1)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
    gmsh.model.mesh.generate(2)
    model = gmsh.model
    gmsh.model.setCurrent("obstacle")
    return model


def build_lshape(resolution):
    gmsh.initialize()
    gmsh.model.add("lshape")
    h = 1.0 / resolution
    big = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
    notch = gmsh.model.occ.addRectangle(0.5, 0.5, 0, 0.5, 0.5)
    cut, _ = gmsh.model.occ.cut([(2, big)], [(2, notch)])
    gmsh.model.occ.synchronize()
    surf_tag = cut[0][1]
    gmsh.model.addPhysicalGroup(2, [surf_tag], tag=1)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
    gmsh.model.mesh.generate(2)
    model = gmsh.model
    gmsh.model.setCurrent("lshape")
    return model


def build_annulus(resolution):
    gmsh.initialize()
    gmsh.model.add("annulus")
    h = 1.0 / resolution
    outer = gmsh.model.occ.addDisk(0, 0, 0, 0.5, 0.5)
    inner = gmsh.model.occ.addDisk(0, 0, 0, 0.2, 0.2)
    cut, _ = gmsh.model.occ.cut([(2, outer)], [(2, inner)])
    gmsh.model.occ.synchronize()
    surf_tag = cut[0][1]
    gmsh.model.addPhysicalGroup(2, [surf_tag], tag=1)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
    gmsh.model.mesh.generate(2)
    model = gmsh.model
    gmsh.model.setCurrent("annulus")
    return model


BUILDERS = {"obstacle": build_obstacle, "lshape": build_lshape, "annulus": build_annulus}


def import_and_solve(name, resolution):
    builder = BUILDERS[name]
    model = builder(resolution)
    mesh_data = dgmsh.model_to_mesh(model, MPI.COMM_WORLD, 0, gdim=2)
    domain = mesh_data.mesh
    gmsh.finalize()

    n_cells = domain.topology.index_map(domain.topology.dim).size_local
    print(f"[{name}] imported mesh: {n_cells} cells, {domain.geometry.x.shape[0]} vertices")
    assert n_cells > 0

    V = fem.functionspace(domain, ("Lagrange", 1))
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, tdim)
    boundary_facets = dmesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, tdim - 1, boundary_facets)
    uD = fem.Function(V)
    uD.x.array[:] = 0.0
    bc = fem.dirichletbc(uD, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(domain, 1.0)
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix=f"spike_gmsh_{name}_{next(_prefix)}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    uh = problem.solve()
    arr = uh.x.array
    assert np.all(np.isfinite(arr))
    print(f"[{name}] solve OK: max={arr.max():.5f} min={arr.min():.5f} ndofs={arr.size}")
    return n_cells


if __name__ == "__main__":
    for name in ("obstacle", "lshape", "annulus"):
        import_and_solve(name, 24)
    print("spike_gmsh: all geometries imported + solved OK")
