"""NavierStokesProcess: a process-bigraph Process wrapping a REAL dolfinx
incompressible Navier-Stokes solve (lid-driven cavity) via the classic IPCS
(Incremental Pressure Correction Scheme -- Chorin-style operator splitting)
used in the dolfinx tutorial's "flow past a cylinder" / lid-driven-cavity
demos. Every substep is a real ``dolfinx.fem.petsc.LinearProblem`` direct
solve (mirrors ``viva_fenics.fem``'s unique-``petsc_options_prefix``-per-solve
pattern) -- there is no reimplementation of the physics and no mock.

Discretization
---------------
Taylor-Hood-ish pair on the unit square: velocity on a **P2 vector**
Lagrange space (``V``), pressure on a **P1** Lagrange space (``Q``). This is
the standard LBB/inf-sup-stable pairing for incompressible Stokes/
Navier-Stokes. Boundary conditions: the top lid (y=1) moves at
``u = (lid_velocity, 0)``; the other three walls are no-slip
(``u = (0, 0)``); pressure is pinned to 0 at a single corner dof (the
bottom-left corner) to remove the additive constant null space that a
fully-Dirichlet-velocity cavity leaves in the pressure Poisson correction
(step 2 below) -- without this the correction system is singular.

Per tick, ``update()`` advances the flow by ``interval`` in substeps of size
``dt`` (config), each substep running the three classic IPCS systems:

1. **Tentative velocity** (``a1``/``L1``): a Crank-Nicolson-in-diffusion,
   explicit-in-convection (using the *previous* step's ``u_n`` for the
   convecting velocity) solve for an intermediate velocity ``u_`` that does
   not yet satisfy the divergence-free constraint.
2. **Pressure correction** (``a2``/``L2``): a Poisson solve for the new
   pressure ``p_`` using ``div(u_)`` as a source, projecting ``u_`` toward a
   divergence-free field.
3. **Velocity correction** (``a3``/``L3``): update the velocity using the
   pressure correction's gradient, producing the new ``u_n`` for the next
   substep/tick.

The three system matrices (``a1``, ``a2``, ``a3``) do not depend on the
solution state (only on mesh/mu/dt, which are fixed once ``config`` is
fixed) -- only the right-hand sides change each substep -- but this module
follows ``viva_fenics.fem``'s established convention of building a fresh
``LinearProblem`` (with a fresh ``petsc_options_prefix``) per solve rather
than caching/reusing an assembled KSP, trading a little performance for
consistency with the rest of the codebase.

This is only *approximately* incompressible: operator splitting introduces
an O(dt) (and, near the lid's two discontinuous BC corners, an O(h) local)
divergence residual that shrinks under mesh/timestep refinement but never
hits exact zero -- see ``divergence_stats`` and the
``navier-stokes``study's ``approximately-divergence-free`` behavior test.
"""

from __future__ import annotations

import itertools

import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc
from process_bigraph import Process

# Unique-ish petsc_options_prefix per solve, same pattern as
# viva_fenics.fem's module-level counter (see that module's docstring for
# why dolfinx 0.11's LinearProblem needs this).
_prefix_counter = itertools.count()


def _build_spaces(resolution):
    """Build the lid-driven-cavity mesh + Taylor-Hood-ish (P2 velocity / P1
    pressure) function space pair.

    Returns:
        (domain, V, Q) tuple.
    """
    domain = mesh.create_unit_square(
        MPI.COMM_WORLD, resolution, resolution,
        diagonal=mesh.DiagonalType.crossed,
    )
    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("Lagrange", 2, (gdim,)))
    Q = fem.functionspace(domain, ("Lagrange", 1))
    return domain, V, Q


def _lid(x):
    return np.isclose(x[1], 1.0)


def _side_and_bottom_walls(x):
    return np.logical_or(
        np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)),
        np.isclose(x[1], 0.0),
    )


def _bottom_left_corner(x):
    return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))


def _lid_driven_cavity_bcs(domain, V, Q, lid_velocity):
    """Dirichlet BCs for the lid-driven cavity: moving lid, no-slip walls,
    and a single pinned pressure dof (see module docstring).
    """
    fdim = domain.topology.dim - 1

    lid_facets = mesh.locate_entities_boundary(domain, fdim, _lid)
    wall_facets = mesh.locate_entities_boundary(domain, fdim, _side_and_bottom_walls)
    lid_dofs = fem.locate_dofs_topological(V, fdim, lid_facets)
    wall_dofs = fem.locate_dofs_topological(V, fdim, wall_facets)

    bc_lid = fem.dirichletbc(PETSc.ScalarType((lid_velocity, 0.0)), lid_dofs, V)
    bc_wall = fem.dirichletbc(PETSc.ScalarType((0.0, 0.0)), wall_dofs, V)

    p_dofs = fem.locate_dofs_geometrical(Q, _bottom_left_corner)
    bc_p = fem.dirichletbc(PETSc.ScalarType(0.0), p_dofs, Q)

    return [bc_lid, bc_wall], [bc_p]


def divergence_stats(domain, V, u_array):
    """Mean/max absolute divergence of a velocity nodal array, interpolated
    onto a DG0 (cell-constant) scalar space.

    A discretization-quality measure of how far an IPCS-split solution is
    from exactly divergence-free -- projection methods are only
    *approximately* incompressible (see module docstring); this is not
    expected to be zero, only small and shrinking under mesh refinement.

    Args:
        domain: dolfinx mesh the velocity field lives on.
        V: the (P2 vector) velocity function space.
        u_array: nodal velocity values (``Function.x.array`` layout).

    Returns:
        (mean_abs_div, max_abs_div) tuple of floats.
    """
    u_fn = fem.Function(V)
    u_fn.x.array[:] = u_array
    DG0 = fem.functionspace(domain, ("DG", 0))
    div_expr = fem.Expression(ufl.div(u_fn), DG0.element.interpolation_points)
    div_field = fem.Function(DG0)
    div_field.interpolate(div_expr)
    abs_div = np.abs(div_field.x.array)
    return float(np.mean(abs_div)), float(np.max(abs_div))


def zero_fields(resolution):
    """Zero-valued (velocity_x, velocity_y, pressure) nodal arrays of the
    correct length for a lid-driven-cavity mesh at this resolution, for
    seeding a composite's stores.

    Never seed a store with ``[]`` -- see
    ``viva_fenics.composites.reaction_diffusion``'s docstring for why an
    empty starting array corrupts the first apply of an ``array[float]``/
    ``overwrite[array[float]]`` store.
    """
    _, V, Q = _build_spaces(resolution)
    n_v = V.tabulate_dof_coordinates().shape[0]
    n_q = Q.tabulate_dof_coordinates().shape[0]
    return (
        np.zeros(n_v, dtype=float),
        np.zeros(n_v, dtype=float),
        np.zeros(n_q, dtype=float),
    )


def velocity_coords(resolution):
    """(N, 2) dof coordinates of the P2 velocity space at this resolution --
    one row per physical velocity node (the block/point coordinate, shared
    by the interleaved x/y components -- see ``NavierStokesProcess``'s
    outputs docstring). NOT the same length as the P1 pressure space's dof
    coordinates.
    """
    _, V, _ = _build_spaces(resolution)
    return V.tabulate_dof_coordinates()[:, :2]


def pressure_coords(resolution):
    """(N, 2) dof coordinates of the P1 pressure space at this resolution.
    Shorter than ``velocity_coords`` at the same resolution (P1 has only
    vertex dofs; P2 also has edge-midpoint dofs).
    """
    _, _, Q = _build_spaces(resolution)
    return Q.tabulate_dof_coordinates()[:, :2]


class NavierStokesProcess(Process):
    """Incompressible lid-driven cavity flow via real dolfinx IPCS splitting.

    Inputs
    ------
    body_force : array[float]
        Additive nodal body-force term (e.g. buoyancy from a sibling
        process); a sibling could wire this in, defaults to zero. NOT YET
        applied to the momentum equation's RHS in this version (the study
        this process ships with runs an unforced cavity) -- accepted here so
        the port exists for a future forced-flow variant without a schema
        change; see the ``navier_stokes`` composite generator for the zero
        default.

    Outputs
    -------
    velocity_x, velocity_y : overwrite[array[float]]
        Current absolute nodal velocity components on the P2 velocity space
        (interleaved x/y layout de-interleaved here into two same-length
        arrays; see ``velocity_coords`` for the matching dof coordinates).
        ``overwrite`` (replace), not a bare additive ``array[float]``: like
        ``DiffusionProcess``'s "integral" sensor, this is the process's own
        internally-persisted absolute flow state (``u_n``), freshly reported
        every tick, with no other process reading *or* writing the same
        store -- an additive apply would accumulate
        ``sum_i velocity_i`` instead of reporting the true current field
        (the same accumulate-vs-overwrite bug class documented on
        ``LogisticReactionProcess.outputs``).
    pressure : overwrite[array[float]]
        Current absolute nodal pressure on the P1 pressure space. Same
        overwrite rationale as velocity_x/velocity_y.
    speed_integral : overwrite[float]
        ``sqrt(integral(dot(u, u)) dx)`` -- the L2 norm of the velocity
        field over the domain, a single scalar sensor reading of overall
        flow intensity (per the sensor rule, ``overwrite[float]``, same
        convention as ``DiffusionProcess.integral``).
    """

    config_schema = {
        "resolution": {"_type": "integer", "_default": 32},
        "reynolds": {"_type": "float", "_default": 100.0},
        "dt": {"_type": "float", "_default": 0.01},
        "lid_velocity": {"_type": "float", "_default": 1.0},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._resolution = None
        self._domain = None
        self._V = None
        self._Q = None
        self._bcu = None
        self._bcp = None
        self._u_n = None  # fem.Function(V), persisted velocity between ticks
        self._p_n = None  # fem.Function(Q), persisted pressure between ticks

    def _ensure_setup(self):
        resolution = self.config["resolution"]
        if self._domain is None or self._resolution != resolution:
            self._domain, self._V, self._Q = _build_spaces(resolution)
            self._bcu, self._bcp = _lid_driven_cavity_bcs(
                self._domain, self._V, self._Q, self.config["lid_velocity"]
            )
            self._u_n = fem.Function(self._V)
            self._p_n = fem.Function(self._Q)
            self._resolution = resolution

    def inputs(self):
        return {"body_force": "array[float]"}

    def outputs(self):
        return {
            "velocity_x": "overwrite[array[float]]",
            "velocity_y": "overwrite[array[float]]",
            "pressure": "overwrite[array[float]]",
            "speed_integral": "overwrite[float]",
        }

    def initial_state(self):
        self._ensure_setup()
        n_v = self._V.tabulate_dof_coordinates().shape[0]
        return {"body_force": np.zeros(2 * n_v)}

    def _ipcs_substep(self, domain, V, Q, bcu, bcp, u_n, p_n, mu, rho, dt):
        """One IPCS substep (tentative velocity -> pressure correction ->
        velocity correction), mutating ``u_n``/``p_n`` in place to the new
        step's fields. See module docstring for the scheme.
        """
        gdim = domain.geometry.dim
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        p = ufl.TrialFunction(Q)
        q = ufl.TestFunction(Q)

        u_ = fem.Function(V)  # tentative velocity
        p_ = fem.Function(Q)  # corrected pressure

        k = fem.Constant(domain, PETSc.ScalarType(dt))
        mu_c = fem.Constant(domain, PETSc.ScalarType(mu))
        rho_c = fem.Constant(domain, PETSc.ScalarType(rho))

        def epsilon(uu):
            return ufl.sym(ufl.nabla_grad(uu))

        def sigma(uu, pp):
            return 2 * mu_c * epsilon(uu) - pp * ufl.Identity(gdim)

        U_avg = 0.5 * (u_n + u)

        # Step 1: tentative velocity.
        F1 = rho_c / k * ufl.dot(u - u_n, v) * ufl.dx
        F1 += rho_c * ufl.dot(ufl.dot(u_n, ufl.nabla_grad(u_n)), v) * ufl.dx
        F1 += ufl.inner(sigma(U_avg, p_n), epsilon(v)) * ufl.dx
        a1 = ufl.lhs(F1)
        L1 = ufl.rhs(F1)
        problem1 = LinearProblem(
            a1, L1, bcs=bcu, u=u_,
            petsc_options_prefix=f"viva_fenics_ns1_{next(_prefix_counter)}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        problem1.solve()

        # Step 2: pressure correction.
        a2 = ufl.dot(ufl.grad(p), ufl.grad(q)) * ufl.dx
        L2 = (
            ufl.dot(ufl.grad(p_n), ufl.grad(q)) * ufl.dx
            - (rho_c / k) * ufl.div(u_) * q * ufl.dx
        )
        problem2 = LinearProblem(
            a2, L2, bcs=bcp, u=p_,
            petsc_options_prefix=f"viva_fenics_ns2_{next(_prefix_counter)}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        problem2.solve()

        # Step 3: velocity correction (solve directly into u_n so it's ready
        # as the convecting field for the next substep).
        a3 = ufl.dot(u, v) * ufl.dx
        L3 = ufl.dot(u_, v) * ufl.dx - k * ufl.dot(ufl.grad(p_ - p_n), v) * ufl.dx
        problem3 = LinearProblem(
            a3, L3, bcs=[], u=u_n,
            petsc_options_prefix=f"viva_fenics_ns3_{next(_prefix_counter)}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        problem3.solve()

        p_n.x.array[:] = p_.x.array

    def update(self, state, interval):
        self._ensure_setup()

        L = 1.0  # unit-square cavity side length
        reynolds = self.config["reynolds"]
        lid_velocity = self.config["lid_velocity"]
        rho = 1.0
        nu = lid_velocity * L / reynolds
        mu = rho * nu

        dt = self.config["dt"]
        n_steps = max(1, round(interval / dt))
        step_dt = interval / n_steps

        for _ in range(n_steps):
            self._ipcs_substep(
                self._domain, self._V, self._Q, self._bcu, self._bcp,
                self._u_n, self._p_n, mu, rho, step_dt,
            )

        velocity_x = self._u_n.x.array[0::2].copy()
        velocity_y = self._u_n.x.array[1::2].copy()
        pressure = self._p_n.x.array.copy()

        speed_l2_form = fem.form(ufl.dot(self._u_n, self._u_n) * ufl.dx)
        speed_integral = float(np.sqrt(
            self._domain.comm.allreduce(fem.assemble_scalar(speed_l2_form), op=MPI.SUM)
        ))

        return {
            "velocity_x": velocity_x,
            "velocity_y": velocity_y,
            "pressure": pressure,
            "speed_integral": speed_integral,
        }
