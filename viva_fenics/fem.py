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
    # dolfinx's default unit-square triangulation (diagonal="right", every
    # quad cut the same way) makes P1 (degree=1) Lagrange elements *nodally
    # exact* for any manufactured solution with constant Laplacian -- i.e.
    # any quadratic, including this module's MMS family -- a well-known
    # structured-mesh superconvergence artifact, NOT genuine O(h^2)
    # discretization behavior. That collapses PoissonSolverStep's l2_error to
    # floating-point round-off (~1e-15) at every resolution for degree=1,
    # masking real convergence rate. "crossed" (4 triangles per quad, no
    # uniform diagonal direction) breaks that artifact -- confirmed
    # empirically: degree=1 error now scales as ~h^2 (rate ~2.0 across
    # resolution doublings) while degree>=2 stays exact to ~1e-13 (unaffected,
    # since a quadratic is exactly representable in the P2+ trial space
    # regardless of triangulation -- Galerkin orthogonality, not a mesh
    # property).
    domain = mesh.create_unit_square(
        MPI.COMM_WORLD, resolution, resolution,
        diagonal=mesh.DiagonalType.crossed,
    )
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

    prefix = f"viva_fenics_poisson_{next(_prefix_counter)}_"
    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix=prefix,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    uh = problem.solve()
    return uh.x.array.copy()


def diffusion_step(domain, V, u_n_array, source_array, dt, D, bc_value=None):
    """Advance one backward-Euler step of the transient diffusion equation.

    Solves for u given the previous nodal field u_n and an additive nodal
    source term:

        (u*v + dt*D*dot(grad(u), grad(v))) dx = (u_n + dt*source) * v dx

    By default no Dirichlet boundary condition is applied, so the boundary is
    natural Neumann (zero-flux) -- mass is conserved up to the source term.

    Args:
        domain: dolfinx mesh.
        V: dolfinx function space over domain.
        u_n_array: nodal values of the field at the start of the step.
        source_array: nodal values of the additive source over the step.
        dt: step size.
        D: diffusion coefficient.
        bc_value: if given, pins the field to this constant value on the
            ``x=0`` face of the domain via a real Dirichlet BC (e.g. a
            morphogen production boundary c=c0), leaving every other face
            natural Neumann (zero-flux). ``None`` (the default) preserves
            the original all-Neumann behavior exactly.

    Returns:
        np.ndarray of nodal solution values after the step (uh.x.array copy).
    """
    u_n = fem.Function(V)
    u_n.x.array[:] = u_n_array

    source = fem.Function(V)
    source.x.array[:] = source_array

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = (u * v + dt * D * ufl.dot(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n + dt * source) * v * ufl.dx

    bcs = []
    if bc_value is not None:
        tdim = domain.topology.dim
        domain.topology.create_connectivity(tdim - 1, tdim)
        boundary_facets = mesh.locate_entities_boundary(
            domain, tdim - 1, lambda x: np.isclose(x[0], 0.0)
        )
        dofs = fem.locate_dofs_topological(V, tdim - 1, boundary_facets)
        bc_fn = fem.Function(V)
        bc_fn.x.array[:] = bc_value
        bcs = [fem.dirichletbc(bc_fn, dofs)]

    prefix = f"viva_fenics_diff_{next(_prefix_counter)}_"
    problem = LinearProblem(
        a, L, bcs=bcs,
        petsc_options_prefix=prefix,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    uh = problem.solve()
    return uh.x.array.copy()


def advection_diffusion_step(
    domain, V, u_n_array, source_array, dt, D,
    velocity=None, soret=None, temperature_array=None,
    bc_value=None, supg=True,
):
    """Advance one backward-Euler step of transient advection-diffusion with an
    optional thermophoretic (Soret) drift term.

    Extends :func:`diffusion_step` -- which solves

        (u*v + dt*D*grad(u).grad(v)) dx = (u_n + dt*source) * v dx

    -- by adding, to the SAME backward-Euler bilinear form, two optional terms
    that leave the ``velocity=None, soret=None`` call bit-for-bit identical to
    ``diffusion_step`` (it is the strict generalisation; see the regression
    test ``test_zero_velocity_matches_diffusion_step``):

    1. **Advection** -- ``+ dt * dot(w, grad(u)) * v dx`` for a divergence-free
       velocity field ``w`` (the Darcy upflow of the vent / pore). With a
       no-penetration (``w.n = 0``) or Dirichlet boundary this conserves the
       field integral up to the source; with ``supg=True`` a
       streamline-upwind Petrov-Galerkin consistent stabilisation is added so
       the solution stays monotone when the cell Peclet number exceeds 1
       (``tau = h / (2|w|)``, the standard high-Peclet limit), instead of the
       spurious node-to-node oscillations a raw Galerkin advection term
       produces.
    2. **Thermophoretic drift (Soret)** -- a conservative thermodiffusive
       flux ``J_T = -D * S_T * u * grad(T)`` (``S_T > 0`` thermophobic, species
       accumulating on the cold side -- the sign the Baaske thermal-trap column
       uses) added in integrated-by-parts (divergence) form,
       ``+ dt * D * S_T * u * grad(T).grad(v) dx``. Because it is the weak
       divergence of a flux, the natural boundary is *zero total flux*
       (``J_T.n = 0``), so the trap redistributes the field (cold-side
       enrichment) while conserving its integral exactly -- the regime the
       og-03 cross-pore trap lives in (see
       ``test_soret_drift_conserves_and_concentrates``).

    Args:
        domain, V: dolfinx mesh + scalar function space (as ``diffusion_step``).
        u_n_array, source_array, dt, D, bc_value: as ``diffusion_step``.
        velocity: the advecting field ``w``. ``None`` (no advection), a
            constant ``(wx, wy)`` tuple (uniform upflow), or a nodal array on
            the vector space ``(V.degree)`` (a resolved flow field, e.g. from
            ``PorousFlowStep``). Length must be ``2 * n_scalar_dofs`` for the
            nodal form.
        soret: Soret coefficient ``S_T`` (1/K). Requires ``temperature_array``.
        temperature_array: nodal temperature field on ``V`` (same layout as
            ``u_n_array``); its gradient drives the drift.
        supg: enable SUPG stabilisation of the advection term (default True).

    Returns:
        np.ndarray of nodal solution values after the step.
    """
    gdim = domain.geometry.dim
    u_n = fem.Function(V)
    u_n.x.array[:] = u_n_array

    source = fem.Function(V)
    source.x.array[:] = source_array

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    dt_c = fem.Constant(domain, float(dt))
    D_c = fem.Constant(domain, float(D))

    # Effective advecting velocity w_eff = w (flow) + w_T (thermophoretic drift).
    # A constant uniform flow is passed as a (wx, wy) tuple/list; a resolved
    # flow field is passed as a nodal numpy array on the vector space.
    w_eff = None
    if velocity is not None:
        degree = V.element.basix_element.degree
        W = fem.functionspace(domain, ("Lagrange", degree, (gdim,)))
        w_fn = fem.Function(W)
        if isinstance(velocity, (tuple, list)):
            comps = [float(c) for c in velocity]
            if len(comps) != gdim:
                raise ValueError(
                    f"constant velocity needs {gdim} components for a {gdim}-D "
                    f"mesh, got {len(comps)}"
                )
            # gdim-general uniform field (was hardcoded to (wx, wy); now 2-D and
            # 3-D both flow through so the same solver drives the 3-D pore mesh).
            w_fn.interpolate(
                lambda x: np.stack([np.full(x.shape[1], c) for c in comps])
            )
        else:
            w_fn.x.array[:] = np.asarray(velocity, dtype=float)
        w_eff = w_fn

    a = (u * v + dt_c * D_c * ufl.dot(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n + dt_c * source) * v * ufl.dx

    # Thermophoretic (Soret) flux J_T = -D*S_T*u*grad(T), added in conservative
    # (integrated-by-parts) form so the natural boundary is zero total flux and
    # the trap conserves the field integral while concentrating it cold-side.
    if soret is not None and temperature_array is not None:
        T_fn = fem.Function(V)
        T_fn.x.array[:] = np.asarray(temperature_array, dtype=float)
        soret_c = fem.Constant(domain, float(soret))
        a += dt_c * D_c * soret_c * u * ufl.dot(ufl.grad(T_fn), ufl.grad(v)) * ufl.dx

    if w_eff is not None:
        a += dt_c * ufl.dot(w_eff, ufl.grad(u)) * v * ufl.dx
        if supg:
            # Transient-consistent SUPG: the strong backward-Euler residual
            # R = (u - u_n) + dt*(w.grad(u)) - dt*source - dt*D*div(grad u)
            # (the last term = 0 for P1) weighted by tau*(w.grad(v)), split into
            # implicit (u, dt*w.grad(u)) and explicit (u_n, dt*source) parts.
            # tau = h/(2|w|) is the standard high-Peclet streamline-upwind
            # scale; note R carries no 1/dt factor -- weighting by 1/dt here
            # over-stabilises by O(1/dt) and corrupts the field at small dt.
            h = ufl.CellDiameter(domain)
            wnorm = ufl.sqrt(ufl.dot(w_eff, w_eff) + 1e-30)
            tau = h / (2.0 * wnorm)
            r_impl = u + dt_c * ufl.dot(w_eff, ufl.grad(u))
            r_expl = u_n + dt_c * source
            a += tau * ufl.dot(w_eff, ufl.grad(v)) * r_impl * ufl.dx
            L += tau * ufl.dot(w_eff, ufl.grad(v)) * r_expl * ufl.dx

    # bc_value: None (all-Neumann), a float (pin x=0 face only, as
    # diffusion_step), or a dict {"x0": v0, "x1": v1} to pin either/both of the
    # x=0 / x=1 faces (the inlet/outlet faces of the vent-pore column, and the
    # advection-diffusion MMS boundary-layer problem).
    bcs = []
    if bc_value is not None:
        tdim = domain.topology.dim
        domain.topology.create_connectivity(tdim - 1, tdim)
        faces = bc_value if isinstance(bc_value, dict) else {"x0": bc_value}
        _locators = {
            "x0": lambda x: np.isclose(x[0], 0.0),
            "x1": lambda x: np.isclose(x[0], 1.0),
        }
        for key, val in faces.items():
            facets = mesh.locate_entities_boundary(domain, tdim - 1, _locators[key])
            dofs = fem.locate_dofs_topological(V, tdim - 1, facets)
            bc_fn = fem.Function(V)
            bc_fn.x.array[:] = val
            bcs.append(fem.dirichletbc(bc_fn, dofs))

    prefix = f"viva_fenics_advdiff_{next(_prefix_counter)}_"
    problem = LinearProblem(
        a, L, bcs=bcs,
        petsc_options_prefix=prefix,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    uh = problem.solve()
    return uh.x.array.copy()


def advection_diffusion_mms_bc(domain, V, wx, D, resolution):
    """Dirichlet BC + analytic solution for the 1-D boundary-layer manufactured
    solution of steady advection-diffusion -D*u'' + wx*u' = 0 on x in [0,1]
    with u(0)=0, u(1)=1 (Neumann top/bottom): u(x) = (exp(wx*x/D)-1)/(exp(wx/D)-1).

    Returns (bc_fn_array, exact_fn) -- ``bc_fn_array`` pins x=0 to 0 and x=1 to
    1 (used via a two-face Dirichlet BC the caller builds), ``exact_fn`` is a
    numpy callable x->u for ``l2_error``.
    """
    Pe = wx / D

    def exact_fn(x):
        return (np.exp(Pe * x[0]) - 1.0) / (np.expm1(Pe))

    return exact_fn


def field_integral(domain, V, array):
    """Integrate a nodal field over the domain: returns a float scalar of
    integral(u) dx, computed via dolfinx's assemble_scalar (consistent with
    the FEM mass matrix, not a naive nodal sum).
    """
    uh = fem.Function(V)
    uh.x.array[:] = array
    form = fem.form(uh * ufl.dx)
    local = fem.assemble_scalar(form)
    return float(domain.comm.allreduce(local, op=MPI.SUM))


def l2_error_exact(domain, V, uh_array, exact_ufl_fn, quadrature_degree=None):
    """Compute the L2 error between a nodal solution and a SYMBOLIC exact
    expression, at elevated quadrature -- for high-order accuracy
    verification, where interpolating the exact solution into the same
    degree-p trial space first (as `l2_error` does) would understate the
    true error: it measures ||uh - I_p(u)|| (distance between two
    degree-p objects) rather than ||uh - u||, silently discarding exactly
    the higher-order information a convergence-rate check needs. Here `u`
    is built as a real UFL expression via `ufl.SpatialCoordinate(domain)`
    (e.g. `ufl.sin`), so the error form can be integrated at a quadrature
    degree well above V's polynomial degree instead.

    Args:
        domain: dolfinx mesh.
        V: dolfinx function space over domain.
        uh_array: nodal values, e.g. returned by solve_poisson.
        exact_ufl_fn: callable(x) -> UFL expression, where x is
            `ufl.SpatialCoordinate(domain)` (NOT a numpy callable like
            `l2_error`'s `exact_fn` -- this one builds symbolic UFL, e.g.
            ``lambda x: ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])``).
        quadrature_degree: quadrature metadata degree; defaults to
            ``2 * (degree + 3)``, comfortably resolving a smooth
            trigonometric manufactured solution without becoming the
            accuracy bottleneck at any of P1/P2/P3.

    Returns:
        L2 norm of (uh - u_exact) as a float.
    """
    uh = fem.Function(V)
    uh.x.array[:] = uh_array

    x = ufl.SpatialCoordinate(domain)
    u_exact = exact_ufl_fn(x)

    degree = V.element.basix_element.degree
    if quadrature_degree is None:
        quadrature_degree = 2 * (degree + 3)
    dx_q = ufl.dx(metadata={"quadrature_degree": quadrature_degree})

    error_form = fem.form((uh - u_exact) ** 2 * dx_q)
    error_local = fem.assemble_scalar(error_form)
    return float(np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM)))


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
