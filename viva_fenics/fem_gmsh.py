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


# ---------------------------------------------------------------------------
# DFG 2D-2 benchmark geometry: channel with an off-center cylinder
# ---------------------------------------------------------------------------

# Schafer/Turek "DFG 2D-2" benchmark dimensions (meters, nondimensional units
# in this solver): a 2.2 x 0.41 channel with a radius-0.05 cylinder centered
# at (0.2, 0.2) -- 0.005 above the channel centerline (0.205), which is what
# breaks top/bottom symmetry and lets the wake shed asymmetrically without
# needing an artificial perturbation.
CHANNEL_LENGTH = 2.2
CHANNEL_HEIGHT = 0.41
CYLINDER_CENTER = (0.2, 0.2)
CYLINDER_RADIUS = 0.05

# Facet-tag marker values used by the imported MeshTags (also the
# `physical_groups` name -> (dim, tag) mapping's tag half) -- exposed so
# ``viva_fenics.processes.flow`` can locate boundary facets by name without
# hardcoding the integers gmsh happened to assign.
CHANNEL_CYLINDER_MARKERS = {"fluid": 1, "inflow": 2, "outflow": 3, "walls": 4, "cylinder": 5}


def _build_channel_cylinder_model(h_cylinder, h_far):
    """Build the DFG 2D-2 channel-with-cylinder gmsh OCC model: a rectangle
    minus a circular disk, with 4 tagged boundary curves (inflow/outflow/
    walls/cylinder) and a graded mesh-size field that refines to
    ``h_cylinder`` near the cylinder and relaxes to ``h_far`` away from it
    (the accuracy driver for resolving vortex shedding in the wake).
    """
    L, H = CHANNEL_LENGTH, CHANNEL_HEIGHT
    cx, cy = CYLINDER_CENTER
    r = CYLINDER_RADIUS

    gmsh.model.add("channel_cylinder")
    rect = gmsh.model.occ.addRectangle(0, 0, 0, L, H)
    disk = gmsh.model.occ.addDisk(cx, cy, 0, r, r)
    cut, _ = gmsh.model.occ.cut([(2, rect)], [(2, disk)])
    gmsh.model.occ.synchronize()
    surf_tag = cut[0][1]
    gmsh.model.addPhysicalGroup(2, [surf_tag], tag=CHANNEL_CYLINDER_MARKERS["fluid"], name="fluid")

    # Classify the 5 boundary curves (4 straight rectangle sides + 1 circle)
    # by bounding box -- robust to whatever tags OCC happened to assign, and
    # to OCC representing a straight side as one curve or several.
    boundary = gmsh.model.getBoundary([(2, surf_tag)], oriented=False)
    inflow, outflow, walls, cylinder = [], [], [], []
    tol = 1e-5
    for dim, tag in boundary:
        xmin, ymin, _zmin, xmax, ymax, _zmax = gmsh.model.getBoundingBox(dim, tag)
        if np.isclose(xmin, 0.0, atol=tol) and np.isclose(xmax, 0.0, atol=tol):
            inflow.append(tag)
        elif np.isclose(xmin, L, atol=tol) and np.isclose(xmax, L, atol=tol):
            outflow.append(tag)
        elif (np.isclose(ymin, 0.0, atol=tol) and np.isclose(ymax, 0.0, atol=tol)) or (
            np.isclose(ymin, H, atol=tol) and np.isclose(ymax, H, atol=tol)
        ):
            walls.append(tag)
        else:
            cylinder.append(tag)

    gmsh.model.addPhysicalGroup(1, inflow, tag=CHANNEL_CYLINDER_MARKERS["inflow"], name="inflow")
    gmsh.model.addPhysicalGroup(1, outflow, tag=CHANNEL_CYLINDER_MARKERS["outflow"], name="outflow")
    gmsh.model.addPhysicalGroup(1, walls, tag=CHANNEL_CYLINDER_MARKERS["walls"], name="walls")
    gmsh.model.addPhysicalGroup(1, cylinder, tag=CHANNEL_CYLINDER_MARKERS["cylinder"], name="cylinder")

    # Graded size field: SizeMin near the cylinder (DistMin=4r), SizeMax far
    # from it (beyond DistMax=15r, which reaches well into the downstream
    # wake where the vortex street forms), linear ramp between. Disable
    # gmsh's default boundary-driven sizing so this field has full control.
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", [float(t) for t in cylinder])
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", h_cylinder)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", h_far)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 4 * r)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 15 * r)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    gmsh.model.mesh.generate(2)


def build_channel_cylinder_mesh(h_cylinder=0.01, h_far=0.06):
    """Build the DFG 2D-2 benchmark channel-with-cylinder mesh and import it
    into dolfinx with tagged boundary facets.

    Args:
        h_cylinder: target mesh element size right at the cylinder boundary
            (the accuracy driver -- must be small enough to resolve the
            boundary layer/shed vortices; DFG-quality accuracy wants
            ``h_cylinder`` around D/10-D/20, i.e. 0.005-0.01).
        h_far: target mesh element size far from the cylinder (channel
            inlet/outlet/walls), coarser to keep the total cell count down.

    Returns:
        (domain, facet_tags, markers) tuple: a real dolfinx ``Mesh`` (from
        the gmsh-generated unstructured triangulation), a ``MeshTags`` over
        exterior facets whose ``.values`` match ``CHANNEL_CYLINDER_MARKERS``,
        and that markers dict itself (``{"inflow": 2, "outflow": 3,
        "walls": 4, "cylinder": 5, "fluid": 1}``) for locating facets by
        name (e.g. ``facet_tags.find(markers["cylinder"])``).
    """
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        _build_channel_cylinder_model(h_cylinder, h_far)
        mesh_data = dgmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
        domain = mesh_data.mesh
        facet_tags = mesh_data.facet_tags
    finally:
        gmsh.finalize()

    return domain, facet_tags, dict(CHANNEL_CYLINDER_MARKERS)


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
