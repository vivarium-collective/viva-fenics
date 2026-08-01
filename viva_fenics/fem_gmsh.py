"""gmsh-generated non-trivial 2D domains imported into dolfinx.

Companion to ``viva_fenics.fem`` (which only builds the trivial
``unit_square`` mesh): this module builds REAL, non-trivial planar
geometries with the ``gmsh`` Python API -- an obstacle domain (unit square
with a circular hole), an L-shaped domain, and an annulus -- and imports
them into a dolfinx ``Mesh`` + ``FunctionSpace`` via
``dolfinx.io.gmsh.model_to_mesh`` (see below for the module-name note),
then solves a real Poisson problem on the imported mesh. Nothing here is
mocked or hand-rolled: gmsh does the actual constructive-solid-geometry +
unstructured triangulation, and dolfinx does the actual FEM assembly/solve
on the resulting mesh.

dolfinx API note
-----------------
Older dolfinx tutorials/docs reference ``dolfinx.io.gmshio``. In the
installed dolfinx 0.11.0 that module was renamed to ``dolfinx.io.gmsh``
(confirmed via ``inspect.signature(dolfinx.io.gmsh.model_to_mesh)`` in
``scratch/spike_gmsh.py``, not from stale memory) and
``model_to_mesh(...)`` now returns a ``MeshData`` ``NamedTuple`` (``.mesh``
is the dolfinx ``Mesh``) rather than a bare ``(mesh, cell_tags,
facet_tags)`` tuple.

gmsh statefulness
------------------
gmsh is a stateful C library (one global "current model" at a time), so
``build_gmsh_mesh`` wraps each build in its own
``gmsh.initialize()``/``gmsh.finalize()`` pair (in a ``try``/``finally``)
so repeated calls -- e.g. building all three geometries in one test run or
one composite sweep -- never leak state across builds.

P1-only cells/coords alignment
-------------------------------
``triangle_cells`` returns triangle vertex-index connectivity from the
mesh TOPOLOGY (``domain.topology.connectivity(tdim, 0)``), which indexes
mesh *vertices*, not general dof numbering. This only lines up with
``node_coords``/a solved field's nodal array for **degree=1** (P1)
Lagrange spaces, where dof numbering equals vertex/geometry numbering
(confirmed empirically: ``V.tabulate_dof_coordinates() ==
domain.geometry.x`` for P1 in ``scratch/spike_gmsh.py``). Do not use
``triangle_cells`` to interpret a degree>=2 solution's node ordering.
"""
from __future__ import annotations

import itertools

import gmsh
import numpy as np
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmsh as dgmsh
from mpi4py import MPI
from petsc4py import PETSc

# Unique-ish petsc_options_prefix per solve -- same convention as
# viva_fenics.fem / viva_fenics.processes.flow / .moving_boundary (dolfinx
# 0.11's LinearProblem needs this; see fem.py's module docstring).
_prefix_counter = itertools.count()

GEOMETRIES = ("obstacle", "lshape", "annulus")


def _build_obstacle_model(resolution):
    """Unit square [0,1]x[0,1] with a circular hole (r=0.2, centered at
    (0.5, 0.5)) cut out -- the classic "flow/diffusion around an obstacle"
    non-trivial domain."""
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


def _build_lshape_model(resolution):
    """L-shaped domain: unit square with its upper-right quadrant
    ([0.5,1]x[0.5,1]) removed -- a classic non-convex re-entrant-corner
    domain (the FEM textbook stress test for corner singularities)."""
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


def _build_annulus_model(resolution):
    """Annulus between r=0.2 (inner) and r=0.5 (outer), centered at the
    origin -- a multiply-connected (has-a-hole) domain."""
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


_BUILDERS = {
    "obstacle": _build_obstacle_model,
    "lshape": _build_lshape_model,
    "annulus": _build_annulus_model,
}


def build_gmsh_mesh(geometry, resolution, degree=1):
    """Build a named gmsh geometry and import it into dolfinx.

    Args:
        geometry: one of ``GEOMETRIES`` ("obstacle" | "lshape" | "annulus").
        resolution: controls the target mesh element size (``h = 1/resolution``,
            passed to gmsh as a per-point target mesh size).
        degree: Lagrange element degree for the returned function space.

    Returns:
        (domain, V) tuple -- a real dolfinx ``Mesh`` (imported from the
        gmsh-generated unstructured triangulation) and a ``FunctionSpace``
        over it.
    """
    if geometry not in _BUILDERS:
        raise ValueError(f"Unsupported geometry: {geometry!r}; choose one of {GEOMETRIES}")

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        _BUILDERS[geometry](resolution)
        mesh_data = dgmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
        domain = mesh_data.mesh
    finally:
        gmsh.finalize()

    V = fem.functionspace(domain, ("Lagrange", degree))
    return domain, V


def solve_poisson_gmsh(domain, V, source_value=1.0):
    """Solve the Poisson equation -div(grad(u)) = source_value with a
    homogeneous Dirichlet BC (u=0) on the FULL exterior boundary of a
    gmsh-imported mesh.

    Unlike ``viva_fenics.fem.solve_poisson`` (which uses a
    manufactured-solution Dirichlet BC with a known closed-form exact
    answer), these gmsh domains have irregular -- and for "annulus",
    multiply-connected -- boundaries with no simple closed-form exact
    solution. This uses the standard textbook "constant unit source, zero
    Dirichlet boundary" setup instead: the solution is a real, non-trivial
    bump that peaks somewhere in the domain interior and is exactly zero on
    every boundary component (including the inner hole boundary for
    "obstacle"/"annulus").

    Args:
        domain: dolfinx mesh (from ``build_gmsh_mesh``).
        V: dolfinx function space over domain.
        source_value: constant right-hand-side source strength.

    Returns:
        np.ndarray of nodal solution values (uh.x.array copy).
    """
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, tdim)
    boundary_facets = dmesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, tdim - 1, boundary_facets)
    uD = fem.Function(V)
    uD.x.array[:] = 0.0
    bc = fem.dirichletbc(uD, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(domain, PETSc.ScalarType(source_value))
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    prefix = f"viva_fenics_gmsh_{next(_prefix_counter)}_"
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


def triangle_cells(domain):
    """(M, 3) triangle vertex-index connectivity of ``domain``.

    Only aligned with a degree=1 (P1) solution's nodal array ordering --
    see module docstring's "P1-only cells/coords alignment" note.
    """
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0)
    c2v = domain.topology.connectivity(tdim, 0)
    return c2v.array.reshape(-1, 3).copy()


def n_cells(domain):
    """Local cell count of the imported mesh (real triangulation, not an
    estimate)."""
    return int(domain.topology.index_map(domain.topology.dim).size_local)
