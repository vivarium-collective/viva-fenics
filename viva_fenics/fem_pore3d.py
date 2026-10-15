"""A representative **3-D** pore geometry (gmsh OCC) imported into dolfinx.

Companion to :mod:`viva_fenics.fem_gmsh` (which builds 2-D planar
geometries) and :mod:`viva_fenics.fem` (the trivial ``unit_square``): this
module builds a real 3-D pore -- a converging-diverging throat between two
chambers, the canonical "hydrothermal pore / mineral vesicle" shape that a
thermophoretic trap concentrates solute in -- with the ``gmsh`` Python API
and imports it into a dolfinx ``Mesh`` (+ tagged boundary facets) via
``dolfinx.io.gmsh.model_to_mesh(..., gdim=3)``.

Nothing is mocked: gmsh does the constructive-solid-geometry (two truncated
cones fused base-to-base into an hourglass) and the unstructured
tetrahedral meshing; dolfinx does the FEM assembly/solve on the result. The
flow axis is **+z**; ``z=0`` is the inlet face, ``z=length`` the outlet, the
conical lateral surface the (no-penetration) pore wall.

Why this geometry
-----------------
The og-03 trap is a 1-D FiPy column / 2-D unit square today; neither has a
real pore wall or a throat. A converging-diverging pore is the smallest 3-D
geometry that makes the physics the full modeling plan (§4.5, Stage III)
asks for non-trivial: the throat sets a local Peclet maximum (the advective
velocity peaks where the cross-section is smallest, by continuity), so the
accumulation boundary layer is thinnest exactly where the mesh must be
finest -- the resolution-sensitivity the "trustworthy Peclet envelope"
measures. The chambers give the solute somewhere to accumulate.

gmsh statefulness
-----------------
gmsh is a stateful C library (one global model), so the builder wraps its
build in its own ``gmsh.initialize()``/``gmsh.finalize()`` pair (in a
``try``/``finally``) so sweeping geometry across a convergence study never
leaks state between builds -- the same discipline as ``fem_gmsh``.
"""
from __future__ import annotations

import gmsh
import numpy as np
from dolfinx.io import gmsh as dgmsh
from mpi4py import MPI

# Facet-tag marker values (mirrors fem_gmsh's convention: 1 = the volume,
# 2/3 = inlet/outlet caps, 4 = the pore wall).
PORE3D_MARKERS = {"pore": 1, "inlet": 2, "outlet": 3, "wall": 4}


def _build_pore3d_model(length, chamber_radius, throat_radius, h_throat, h_chamber):
    """Build the hourglass pore in the CURRENT gmsh model (caller owns
    initialize/finalize). Two truncated cones fused base-to-base: a converging
    cone chamber_radius -> throat_radius over [0, length/2] and a diverging cone
    throat_radius -> chamber_radius over [length/2, length], flow along +z."""
    occ = gmsh.model.occ
    half = length / 2.0
    # addCone(x, y, z, dx, dy, dz, r1, r2): truncated cone from radius r1 at the
    # base (x,y,z) to r2 at the top (x+dx, y+dy, z+dz).
    cone_in = occ.addCone(0, 0, 0, 0, 0, half, chamber_radius, throat_radius)
    cone_out = occ.addCone(0, 0, half, 0, 0, half, throat_radius, chamber_radius)
    fused, _ = occ.fuse([(3, cone_in)], [(3, cone_out)])
    occ.synchronize()

    vol_tag = fused[0][1]
    gmsh.model.addPhysicalGroup(3, [vol_tag], tag=PORE3D_MARKERS["pore"], name="pore")

    # Classify the boundary surfaces by their centre of mass: z~=0 -> inlet,
    # z~=length -> outlet, everything else -> the conical wall.
    inlet, outlet, wall = [], [], []
    for dim, surf in gmsh.model.getBoundary([(3, vol_tag)], oriented=False):
        if dim != 2:
            continue
        _, _, zc = occ.getCenterOfMass(2, surf)
        if np.isclose(zc, 0.0, atol=1e-6 * length):
            inlet.append(surf)
        elif np.isclose(zc, length, atol=1e-6 * length):
            outlet.append(surf)
        else:
            wall.append(surf)
    gmsh.model.addPhysicalGroup(2, inlet, tag=PORE3D_MARKERS["inlet"], name="inlet")
    gmsh.model.addPhysicalGroup(2, outlet, tag=PORE3D_MARKERS["outlet"], name="outlet")
    gmsh.model.addPhysicalGroup(2, wall, tag=PORE3D_MARKERS["wall"], name="wall")

    # Graded sizing: fine at the throat (z = length/2, the thinnest cross
    # section and highest local Peclet), coarse toward the chambers. Distance
    # from the throat plane drives a Threshold field.
    probe = gmsh.model.occ.addPoint(0, 0, half)
    occ.synchronize()
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "PointsList", [probe])
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", h_throat)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", h_chamber)
    gmsh.model.mesh.field.setNumber(2, "DistMin", throat_radius)
    gmsh.model.mesh.field.setNumber(2, "DistMax", half)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.model.mesh.generate(3)


def build_pore3d_mesh(
    length=1.0,
    chamber_radius=0.4,
    throat_radius=0.12,
    h_throat=0.03,
    h_chamber=0.09,
):
    """Build a representative 3-D converging-diverging pore and import it into
    dolfinx.

    Args:
        length: pore length along the flow (+z) axis (nondimensional units).
        chamber_radius: radius at the inlet/outlet caps.
        throat_radius: radius at the mid-plane constriction (< chamber_radius).
        h_throat: target mesh size at the throat (the fine region).
        h_chamber: target mesh size in the chambers (the coarse region).

    Returns:
        ``(domain, facet_tags, markers)``: a real dolfinx 3-D ``Mesh`` (from
        gmsh's tetrahedral mesh), a ``MeshTags`` on exterior facets whose
        ``.values`` match :data:`PORE3D_MARKERS`, and a copy of that marker
        dict. Select a boundary by name, e.g. ``facet_tags.find(markers["inlet"])``.
    """
    if throat_radius >= chamber_radius:
        raise ValueError("throat_radius must be smaller than chamber_radius")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        _build_pore3d_model(length, chamber_radius, throat_radius,
                            h_throat, h_chamber)
        mesh_data = dgmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
        domain = mesh_data.mesh
        facet_tags = mesh_data.facet_tags
    finally:
        gmsh.finalize()
    return domain, facet_tags, dict(PORE3D_MARKERS)


def pore_axis_coords(V):
    """Nodal coordinates of ``V`` as an ``(N, 3)`` array (the flow axis is
    column 2, ``z``). Thin wrapper over the dof coordinates for readouts that
    bin along the pore axis (inlet->throat->outlet enrichment profiles)."""
    return V.tabulate_dof_coordinates()[:, :3]
