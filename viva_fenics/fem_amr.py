"""Residual-based Adaptive Mesh Refinement (AMR) on the classic L-shaped
domain re-entrant-corner singularity.

The problem
-----------
``Omega = (-1,1) x (-1,1) \\ [0,1] x [-1,0]`` -- the unit square with its
lower-right quadrant removed, leaving a re-entrant (interior angle
``omega = 3*pi/2``, i.e. 270 degrees) corner at the ORIGIN. This is a
*different* L-shape than ``fem_gmsh``'s ``"lshape"`` geometry (a unit square
minus its upper-right quadrant, corner at ``(0.5, 0.5)``) -- this module
needs the corner AT THE ORIGIN to line up with the closed-form singular
harmonic function below, so it builds its own gmsh model rather than reusing
``fem_gmsh``.

Laplace's equation (``f=0``) on this domain, in polar coordinates centered
at the re-entrant corner, has the classic singular harmonic solution

    u(r, theta) = r^(2/3) * sin(2*theta/3),   theta in [0, 3*pi/2)

(``theta`` measured counterclockwise from the positive x-axis, wrapped into
``[0, 2*pi)`` so it is single-valued and continuous across the domain's
interior -- the ``[3*pi/2, 2*pi)`` range corresponds to the removed
quadrant and is never evaluated). This is the standard re-entrant-corner
benchmark (e.g. Mitchell 2013's "L-shaped domain" test problem; the same
closed form used in FEniCS/deal.II AMR tutorials): ``u`` is harmonic away
from the origin, vanishes at the origin and on both straight edges forming
the corner (``theta=0`` and ``theta=3*pi/2``), and its gradient blows up as
``r^(-1/3)`` at the corner -- an INTEGRABLE singularity (``|grad u|^2 ~
r^(-2/3)``, and ``integral r^(-2/3) * r dr`` converges), so the exact energy
norm is finite, but the singularity limits the solution's regularity to
``H^(1+2/3-eps)``, which is exactly what caps UNIFORM refinement's energy-norm
convergence rate at ``O(h^(2/3))`` (i.e. ``O(dofs^(-1/3))`` in 2D) instead of
the usual ``O(h)`` (``O(dofs^(-1/2))``) for a smooth solution -- see
``solve_laplace_lshape``'s docstring for the a priori theory.

The Dirichlet BC is this SAME exact solution imposed on the FULL domain
boundary (not just the two corner edges) -- both because it is the natural
thing to do (any harmonic function is fully determined by its boundary
values) and because it makes the energy-norm error against ``u`` directly
computable for validation, exactly like ``viva_fenics.fem``'s
manufactured-solution convention.

The estimator
-------------
``error_indicators`` implements the standard (Verfurth) residual-based a
posteriori error estimator for ``-Laplacian(u) = f`` with P_k conforming
elements:

    eta_T^2 = h_T^2 * ||f + Laplacian(u_h)||^2_{L2(T)}
              + (1/2) * sum_{e in interior(T)} h_e * ||[[du_h/dn]]||^2_{L2(e)}

For P1 (the default here) ``u_h`` is piecewise LINEAR, so
``Laplacian(u_h) == 0`` pointwise on every cell; with ``f=0`` (Laplace, not
Poisson) the interior "cell residual" term is identically zero and the
estimator collapses to the edge-jump term alone -- the jump in the normal
derivative of ``u_h`` across each interior edge, weighted by the edge
length. This term is exactly what should spike near the re-entrant corner:
``u_h``'s gradient there is trying to approximate a genuinely unbounded
(``r^(-1/3)``) exact gradient, so adjacent cells' constant gradients
disagree sharply -- ``dorfler_mark`` then concentrates refinement exactly
where that disagreement (the jump) is largest.

Real dolfinx (0.11.0) throughout -- no faking. Verified against
``scratch/spike_amr.py``.
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

# Unique-ish petsc_options_prefix per solve -- same convention as
# viva_fenics.fem / viva_fenics.fem_gmsh (dolfinx 0.11's LinearProblem needs
# this to avoid clashing PETSc option namespaces across repeated solves in
# one process -- an AMR loop calls solve_laplace_lshape many times).
_prefix_counter = itertools.count()

# The re-entrant corner and its interior angle -- documented for callers
# that want to compute distance-from-corner (e.g. checking that refinement
# concentrates there) without hardcoding the origin twice.
REENTRANT_CORNER = (0.0, 0.0)
CORNER_ANGLE = 1.5 * np.pi  # 270 degrees

# Theoretical energy-norm convergence rates (error ~ dofs^rate) for this
# problem in 2D, P1 elements -- see module docstring. Exposed so callers
# (tests, run.py) compare achieved rates against theory without
# re-deriving/hardcoding the -1/3 and -1/2 magic numbers independently.
UNIFORM_RATE_THEORY = -1.0 / 3.0
ADAPTIVE_RATE_THEORY = -1.0 / 2.0


def build_lshape_amr_mesh(h):
    """Build the AMR benchmark L-shaped domain and import it into dolfinx.

    ``Omega = [-1,1]^2 \\ [0,1]x[-1,0]`` (re-entrant corner at the origin,
    see module docstring) -- a real gmsh OCC boolean cut, unstructured
    triangulation, imported via ``dolfinx.io.gmsh.model_to_mesh`` exactly
    like ``fem_gmsh``'s own mesh builders (e.g.
    ``fem_gmsh.build_channel_cylinder_mesh``).

    Args:
        h: target mesh element size passed to gmsh (coarser initial mesh ->
            more AMR levels needed to resolve the corner; this is the
            STARTING mesh for a refinement sequence, not a fixed resolution).

    Returns:
        A real dolfinx ``Mesh``.
    """
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


def exact_solution_ufl(domain):
    """UFL symbolic expression for the exact singular harmonic solution
    ``u(r,theta) = r^(2/3) * sin(2*theta/3)`` over ``domain``'s spatial
    coordinate.

    Built with ``ufl.atan2``/``ufl.sqrt`` (real UFL AD, not a discretized
    numpy approximation) so ``ufl.grad(...)`` of this expression gives the
    EXACT symbolic gradient -- used both to build the Dirichlet BC (via
    ``fem.Expression`` interpolation) and to compute the true energy-norm
    error directly against the analytic solution (not against a nodally
    -interpolated, and therefore itself-erroneous-near-the-corner,
    discretization of it).
    """
    x = ufl.SpatialCoordinate(domain)
    r = ufl.sqrt(x[0] ** 2 + x[1] ** 2)
    theta = ufl.atan2(x[1], x[0])
    theta = ufl.conditional(ufl.lt(theta, 0), theta + 2 * ufl.pi, theta)
    return r ** (2.0 / 3.0) * ufl.sin(2.0 / 3.0 * theta)


def solve_laplace_lshape(domain, degree=1):
    """Solve ``-Laplacian(u) = 0`` on ``domain`` with the exact singular
    solution imposed as the Dirichlet BC on the full boundary, and compute
    the H1-seminorm (energy-norm) error against that exact solution.

    A priori theory (see module docstring): the exact solution has a
    ``r^(2/3)`` singularity at the re-entrant corner, limiting its Sobolev
    regularity to just under ``H^(5/3)``. Standard a priori FEM error
    theory then gives, for P1 elements:

    - UNIFORM refinement: energy error ~ h^(2/3) ~ dofs^(-1/3) in 2D
      (regularity-limited, NOT the usual O(h) ~ dofs^(-1/2) for a smooth
      solution -- refining uniformly wastes most new dofs far from the
      corner, where the solution is already well-resolved).
    - Elements OPTIMALLY graded toward the corner (what adaptive
      refinement, if it works, should discover on its own via the error
      estimator): energy error ~ dofs^(-1/2), the optimal rate for P1
      recovered despite the singularity.

    Args:
        domain: dolfinx mesh (e.g. from ``build_lshape_amr_mesh`` or
            ``refine_marked``/``refine_uniform``).
        degree: Lagrange element degree (P1 by default -- see module
            docstring for why the estimator's cell-residual term vanishes
            specifically at degree=1).

    Returns:
        (uh, V, energy_error): the solved ``dolfinx.fem.Function``, its
        function space, and the float H1-seminorm error
        ``sqrt(integral |grad(uh) - grad(u_exact)|^2 dx)``, integrated with
        an elevated quadrature degree (the integrand involves the exact
        solution's non-polynomial, near-singular gradient, which UFL's
        automatic quadrature-degree estimation does not handle well).
    """
    V = fem.functionspace(domain, ("Lagrange", degree))
    u_exact = exact_solution_ufl(domain)

    uD = fem.Function(V)
    bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    uD.interpolate(bc_expr)

    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, tdim)
    boundary_facets = dmesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, tdim - 1, boundary_facets)
    bc = fem.dirichletbc(uD, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = fem.Constant(domain, 0.0) * v * ufl.dx  # f = 0 (Laplace, not Poisson)

    prefix = f"viva_fenics_amr_{next(_prefix_counter)}_"
    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix=prefix,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    uh = problem.solve()

    dx_elevated = ufl.dx(metadata={"quadrature_degree": 12})
    diff = ufl.grad(uh) - ufl.grad(u_exact)
    err_form = fem.form(ufl.dot(diff, diff) * dx_elevated)
    err_local = fem.assemble_scalar(err_form)
    energy_error = float(np.sqrt(domain.comm.allreduce(err_local, op=MPI.SUM)))

    return uh, V, energy_error


def error_indicators(domain, uh):
    """Per-cell residual-based a posteriori error indicators (squared),
    ``eta_T^2``, for the solved field ``uh`` on ``domain`` -- see module
    docstring for the estimator formula (edge-jump term only, since the
    interior cell-residual term vanishes for P1 with ``f=0``).

    Returns:
        np.ndarray, shape (n_local_cells,), of ``eta_T^2`` values indexed by
        DG0 cell/dof number (DG0's dof numbering equals cell numbering).
    """
    DG0 = fem.functionspace(domain, ("DG", 0))
    w = ufl.TestFunction(DG0)
    n = ufl.FacetNormal(domain)
    he = ufl.FacetArea(domain)  # edge length, in 2D
    jump_sq = ufl.jump(ufl.grad(uh), n) ** 2
    # avg(w) distributes half of each interior edge's contribution to each
    # of its two adjacent cells' DG0 dof -- assembling this linear form
    # directly produces the per-cell eta_T^2 sum (no boundary-facet term:
    # correct here since the Dirichlet data is exact, not approximated).
    eta_form = fem.form(he("+") * ufl.avg(w) * jump_sq * ufl.dS)
    eta_vec = fem.assemble_vector(eta_form)
    return eta_vec.array.copy()


def dorfler_mark(indicators, theta=0.5):
    """Doerfler (bulk-chasing) marking: the SMALLEST set of cells whose
    summed squared indicators reach a ``theta`` fraction of the total
    squared estimated error.

    Args:
        indicators: per-cell eta_T^2 array (from ``error_indicators``).
        theta: target fraction of total estimated error to capture
            (0 < theta <= 1); larger theta marks more cells per level
            (faster dof growth, fewer levels needed).

    Returns:
        Sorted np.ndarray (int32) of marked cell indices.
    """
    indicators = np.asarray(indicators, dtype=float)
    total = float(indicators.sum())
    if total <= 0.0 or indicators.size == 0:
        return np.array([], dtype=np.int32)
    order = np.argsort(indicators)[::-1]
    cumulative = np.cumsum(indicators[order])
    cutoff = int(np.searchsorted(cumulative, theta * total)) + 1
    cutoff = min(cutoff, indicators.size)
    return np.sort(order[:cutoff]).astype(np.int32)


def refine_marked(domain, marked_cells):
    """Conformingly refine ``domain`` so every edge of every cell in
    ``marked_cells`` is bisected.

    dolfinx's ``mesh.refine(msh, edges=...)`` takes a set of EDGES to split
    (not cells), and internally propagates extra splits beyond the given
    set as needed to keep the refined mesh conforming (no hanging nodes) --
    the standard Plaza/NVB algorithm. So marking cells for refinement means
    marking the UNION of their edges first, via
    ``mesh.compute_incident_entities``.

    Args:
        domain: dolfinx mesh.
        marked_cells: int array of local cell indices to refine (e.g. from
            ``dorfler_mark``).

    Returns:
        The refined dolfinx ``Mesh``.
    """
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 1)
    domain.topology.create_connectivity(1, tdim)
    marked_edges = dmesh.compute_incident_entities(
        domain.topology, np.asarray(marked_cells, dtype=np.int32), tdim, 1
    )
    marked_edges = np.unique(marked_edges).astype(np.int32)
    new_domain, _parent_cells, _parent_facets = dmesh.refine(domain, edges=marked_edges)
    return new_domain


def refine_uniform(domain):
    """Uniformly refine every cell of ``domain`` (``mesh.refine`` with
    ``edges=None``): every edge is bisected, quadrupling the 2D cell count.
    """
    # dolfinx.mesh.refine needs the edge (dim=1) IndexMap to already exist
    # on the topology; solve_laplace_lshape creates it as a side effect (via
    # create_connectivity(tdim-1, tdim)), but a caller refining a mesh that
    # was never solved on first (e.g. straight after
    # build_lshape_amr_mesh) hits "Missing IndexMap in Topology" without
    # this explicit create_entities call.
    domain.topology.create_entities(1)
    new_domain, _parent_cells, _parent_facets = dmesh.refine(domain)
    return new_domain


def node_coords(V):
    """(N, 2) dof coordinates for a P1 function space V."""
    return V.tabulate_dof_coordinates()[:, :2]


def triangle_cells(domain):
    """(M, 3) triangle vertex-index connectivity of ``domain``.

    Only aligned with a P1 solution's nodal array ordering -- see
    ``fem_gmsh``'s module docstring's "P1-only cells/coords alignment" note
    (same convention, same caveat, reused here).
    """
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0)
    c2v = domain.topology.connectivity(tdim, 0)
    return c2v.array.reshape(-1, 3).copy()


def n_cells(domain):
    """Local cell count of ``domain`` (real triangulation, not an estimate)."""
    return int(domain.topology.index_map(domain.topology.dim).size_local)


def run_amr_loop(initial_h, n_refinements, theta=0.5, degree=1, capture_levels=None):
    """Run the full residual-based Doerfler-marking AMR loop from a fresh
    ``build_lshape_amr_mesh(initial_h)`` mesh.

    Each level: solve -> compute indicators -> Doerfler-mark -> refine
    (except the last level, which only solves -- so ``n_refinements + 1``
    solves total for ``n_refinements`` refinement steps).

    Args:
        initial_h: starting mesh element size (``build_lshape_amr_mesh``).
        n_refinements: number of REFINEMENT steps (levels 0..n_refinements
            inclusive are solved, i.e. n_refinements+1 solves).
        theta: Doerfler marking fraction (``dorfler_mark``).
        degree: Lagrange element degree.
        capture_levels: optional iterable of level indices at which to also
            snapshot the mesh geometry (node coords + triangle cells) for
            visualization -- e.g. an animation showing refinement
            concentrating at the corner. Omit (default) to skip snapshotting
            entirely (cheaper: no geometry extraction on levels that don't
            need it).

    Returns:
        dict with:
        - "dofs_history": np.ndarray, global dof count at each level.
        - "error_history": np.ndarray, energy-norm error at each level.
        - "uh", "V", "domain": the FINAL level's solution/space/mesh.
        - "snapshots": {level: (coords, cells)} for each requested
          ``capture_levels`` entry that was actually reached.
    """
    domain = build_lshape_amr_mesh(initial_h)
    capture = set(capture_levels or ())
    dofs_history: list[float] = []
    error_history: list[float] = []
    snapshots = {}
    uh = V = None

    for level in range(n_refinements + 1):
        uh, V, energy_error = solve_laplace_lshape(domain, degree=degree)
        dofs_history.append(float(V.dofmap.index_map.size_global))
        error_history.append(energy_error)

        if level in capture:
            snapshots[level] = (node_coords(V), triangle_cells(domain))

        if level == n_refinements:
            break

        indicators = error_indicators(domain, uh)
        marked = dorfler_mark(indicators, theta=theta)
        domain = refine_marked(domain, marked)

    return {
        "dofs_history": np.array(dofs_history),
        "error_history": np.array(error_history),
        "uh": uh,
        "V": V,
        "domain": domain,
        "snapshots": snapshots,
    }


def run_uniform_loop(initial_h, n_levels, degree=1):
    """Run a uniform-refinement sequence from a fresh
    ``build_lshape_amr_mesh(initial_h)`` mesh, for comparison against
    ``run_amr_loop``'s adaptive sequence.

    Args:
        initial_h: starting mesh element size.
        n_levels: number of uniform REFINEMENT steps (n_levels+1 solves).
        degree: Lagrange element degree.

    Returns:
        dict with "dofs_history", "error_history", "uh", "V", "domain"
        (same shape as ``run_amr_loop``, minus "snapshots").
    """
    domain = build_lshape_amr_mesh(initial_h)
    dofs_history: list[float] = []
    error_history: list[float] = []
    uh = V = None

    for level in range(n_levels + 1):
        uh, V, energy_error = solve_laplace_lshape(domain, degree=degree)
        dofs_history.append(float(V.dofmap.index_map.size_global))
        error_history.append(energy_error)
        if level == n_levels:
            break
        domain = refine_uniform(domain)

    return {
        "dofs_history": np.array(dofs_history),
        "error_history": np.array(error_history),
        "uh": uh,
        "V": V,
        "domain": domain,
    }
