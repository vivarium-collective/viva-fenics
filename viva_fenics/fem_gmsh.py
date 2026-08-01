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


# ---------------------------------------------------------------------------
# Porous lattice geometry: channel with a periodic array of circular pillars
# ---------------------------------------------------------------------------

# Unit channel: a 1x1 square domain tiled with an nx-by-ny grid of circular
# pillars -- one "unit cell" per grid point, spacing = L/nx by H/ny -- so the
# whole channel is packed edge-to-edge with the lattice (no inlet/outlet
# buffer), the standard finite-window approximation of an infinite periodic
# porous medium used to estimate its effective (Darcy) permeability.
POROUS_DOMAIN_LENGTH = 1.0
POROUS_DOMAIN_HEIGHT = 1.0

# Facet-tag marker values (mirrors CHANNEL_CYLINDER_MARKERS's convention --
# "pillars" replaces the single-obstacle "cylinder" group since there are
# nx*ny disks here, all sharing one physical-group tag).
POROUS_LATTICE_MARKERS = {"fluid": 1, "inflow": 2, "outflow": 3, "walls": 4, "pillars": 5}


def lattice_porosity(nx, ny, pillar_radius, length=POROUS_DOMAIN_LENGTH, height=POROUS_DOMAIN_HEIGHT):
    """Analytic (geometry-formula) porosity of an nx-by-ny circular-pillar
    lattice: 1 - (total pillar area) / (channel area).

    A quick closed-form estimate for sizing/sanity purposes -- the
    authoritative porosity used by the Stokes solve is the FEM-ASSEMBLED
    fluid-area fraction returned by ``build_porous_lattice_mesh`` (the real
    meshed domain, not this polygon-free idealization; gmsh's triangulated
    circle boundary makes the two agree only approximately, closer as the
    mesh is refined).
    """
    pillar_area = nx * ny * np.pi * pillar_radius**2
    return 1.0 - pillar_area / (length * height)


def _build_porous_lattice_model(nx, ny, pillar_radius, h_pillar, h_far):
    """Build the porous-lattice gmsh OCC model: a unit channel rectangle
    minus an nx-by-ny grid of circular pillars, with 4 tagged boundary
    groups (inflow/outflow/walls/pillars) and a graded mesh-size field that
    refines to ``h_pillar`` at every pillar boundary and relaxes to
    ``h_far`` away from all of them.
    """
    L, H = POROUS_DOMAIN_LENGTH, POROUS_DOMAIN_HEIGHT
    spacing_x, spacing_y = L / nx, H / ny
    max_r = 0.5 * min(spacing_x, spacing_y)
    if pillar_radius >= max_r:
        raise ValueError(
            f"pillar_radius={pillar_radius} too large for a {nx}x{ny} lattice "
            f"(spacing={spacing_x:.4g}x{spacing_y:.4g}); must be < {max_r:.4g} "
            "to avoid touching/overlapping neighboring pillars or the channel wall"
        )

    gmsh.model.add("porous_lattice")
    rect = gmsh.model.occ.addRectangle(0, 0, 0, L, H)
    pillar_dimtags = []
    for i in range(nx):
        for j in range(ny):
            cx = (i + 0.5) * spacing_x
            cy = (j + 0.5) * spacing_y
            disk = gmsh.model.occ.addDisk(cx, cy, 0, pillar_radius, pillar_radius)
            pillar_dimtags.append((2, disk))
    cut, _ = gmsh.model.occ.cut([(2, rect)], pillar_dimtags)
    gmsh.model.occ.synchronize()
    if len(cut) != 1:
        raise RuntimeError(
            f"expected the pillar lattice cut to leave a single connected fluid "
            f"surface, got {len(cut)} -- pillars may be touching/splitting the domain"
        )
    surf_tag = cut[0][1]
    gmsh.model.addPhysicalGroup(2, [surf_tag], tag=POROUS_LATTICE_MARKERS["fluid"], name="fluid")

    # Classify boundary curves by bounding box, same approach as
    # _build_channel_cylinder_model: the 4 straight channel sides go to
    # inflow/outflow/walls, every circular pillar boundary (not aligned with
    # any straight edge) falls through to "pillars".
    boundary = gmsh.model.getBoundary([(2, surf_tag)], oriented=False)
    inflow, outflow, walls, pillars = [], [], [], []
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
            pillars.append(tag)

    gmsh.model.addPhysicalGroup(1, inflow, tag=POROUS_LATTICE_MARKERS["inflow"], name="inflow")
    gmsh.model.addPhysicalGroup(1, outflow, tag=POROUS_LATTICE_MARKERS["outflow"], name="outflow")
    gmsh.model.addPhysicalGroup(1, walls, tag=POROUS_LATTICE_MARKERS["walls"], name="walls")
    gmsh.model.addPhysicalGroup(1, pillars, tag=POROUS_LATTICE_MARKERS["pillars"], name="pillars")

    # Graded size field: SizeMin (h_pillar) right at every pillar boundary,
    # SizeMax (h_far) away from all of them -- same Distance/Threshold
    # pattern as _build_channel_cylinder_model, but the CurvesList spans
    # ALL nx*ny pillar boundaries at once so every pillar-pillar gap in the
    # lattice gets resolved, not just the near-field of a single obstacle.
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", [float(t) for t in pillars])
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", h_pillar)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", h_far)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 2 * pillar_radius)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 4 * pillar_radius)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    gmsh.model.mesh.generate(2)


def build_porous_lattice_mesh(nx=4, ny=4, pillar_radius=0.08, h_pillar=0.015, h_far=0.05):
    """Build a periodic-pillar porous-lattice mesh (a unit channel packed
    with an nx-by-ny grid of circular pillars) and import it into dolfinx
    with tagged boundary facets, for a Stokes flow -> effective-permeability
    study (see ``viva_fenics.processes.flow.PorousFlowStep``).

    Args:
        nx, ny: pillar grid dimensions.
        pillar_radius: pillar disk radius -- the porosity-control knob (must
            be < half the lattice spacing to avoid pillar-pillar/pillar-wall
            overlap; see ``_build_porous_lattice_model``).
        h_pillar: target mesh element size right at every pillar boundary
            (the accuracy driver for resolving the boundary-layer flow
            through the pore throats between adjacent pillars).
        h_far: target mesh element size away from the pillars (channel
            inlet/outlet/walls).

    Returns:
        (domain, facet_tags, markers, porosity) tuple: a real dolfinx
        ``Mesh`` imported from the gmsh triangulation, a ``MeshTags`` over
        exterior facets whose ``.values`` match ``POROUS_LATTICE_MARKERS``,
        that markers dict itself, and the FEM-ASSEMBLED fluid-area fraction
        (``integral(1) dx`` over the imported mesh, divided by the channel's
        total L*H area) -- a real reading off the actual meshed domain, not
        the ``lattice_porosity`` closed-form estimate (see that function's
        docstring for why they only approximately agree).
    """
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        _build_porous_lattice_model(nx, ny, pillar_radius, h_pillar, h_far)
        mesh_data = dgmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
        domain = mesh_data.mesh
        facet_tags = mesh_data.facet_tags
    finally:
        gmsh.finalize()

    one = fem.Constant(domain, PETSc.ScalarType(1.0))
    area_form = fem.form(one * ufl.dx)
    area = float(domain.comm.allreduce(fem.assemble_scalar(area_form), op=MPI.SUM))
    porosity = area / (POROUS_DOMAIN_LENGTH * POROUS_DOMAIN_HEIGHT)

    return domain, facet_tags, dict(POROUS_LATTICE_MARKERS), porosity


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
