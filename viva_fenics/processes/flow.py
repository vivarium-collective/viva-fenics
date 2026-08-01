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

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc
from process_bigraph import Process, Step

from viva_fenics import fem_gmsh

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


# =============================================================================
# CylinderFlowProcess: DFG 2D-2 benchmark -- von Karman vortex street
# =============================================================================
#
# Same IPCS scheme as NavierStokesProcess (see that class's docstring for the
# 3-substep operator-splitting derivation), on a different, harder domain: a
# gmsh-generated channel with an off-center cylindrical obstacle
# (``viva_fenics.fem_gmsh.build_channel_cylinder_mesh``) instead of the unit
# square, with tagged inflow/outflow/walls/cylinder boundaries driving 4
# distinct boundary conditions instead of the cavity's uniform-lid BC, plus a
# real surface-force integral over the cylinder boundary for drag/lift.
#
# This is the DFG 2D-2 (Schafer/Turek) unsteady benchmark: a parabolic
# inflow (mean U=1, peak U_m=1.5) past a radius-0.05 cylinder centered
# (0.2, 0.2) in a 2.2x0.41 channel at nu=1e-3 (Re = U*D/nu = 100). At this
# Reynolds number and off-center placement, the wake is unstable and sheds
# alternating vortices (a von Karman vortex street) -- the flow never
# reaches a steady state; drag oscillates around a mean and lift oscillates
# symmetrically about zero once shedding is established, at the Strouhal
# frequency (St = f*D/U ~ 0.30 at Re=100 per the DFG reference values).
#
# Inflow-ramp note
# -----------------
# An impulsive full-velocity inflow BC at t=0 (from a zero initial
# condition) creates a startup transient large enough that this explicit-
# convection IPCS scheme can go unstable at the dt/mesh combination needed
# to resolve shedding (confirmed empirically during development: mesh/dt
# choices that are otherwise stable blew up (NaN) within the first fraction
# of a second without a ramp). ``inflow_ramp_time`` linearly ramps the
# inflow peak velocity from 0 to ``u_peak`` over that many simulated
# seconds, matching standard practice in DFG-benchmark implementations, and
# does not change the (ramp-independent) coefficient normalization -- see
# ``drag_coeff``/``lift_coeff`` below.
class CylinderFlowProcess(Process):
    """Incompressible flow past a cylinder in a channel (DFG 2D-2 benchmark)
    via real dolfinx IPCS on a gmsh-generated, cylinder-refined mesh --
    produces a von Karman vortex street at Re=100.

    Inputs
    ------
    inflow_perturbation : float
        Additive offset (m/s) on top of the (ramped) parabolic inflow
        peak velocity each substep. Defaults to 0 (the composite this
        process ships with wires a constant zero), i.e. the inflow follows
        the DFG benchmark's prescribed profile exactly; a sibling process
        could inject a controlled disturbance here without a schema change.
        This is what justifies ``CylinderFlowProcess`` being a stateful
        ``Process`` with a real (if usually-zero) input port, per the
        Port-Design "no inputs => Step" convention documented on
        ``PoissonSolverStep``.

    Outputs
    -------
    velocity_x, velocity_y : overwrite[array[float]]
        Current absolute nodal velocity components on the P2 velocity
        space. ``overwrite`` (not additive) for the same reason as
        ``NavierStokesProcess.outputs`` -- see that docstring.
    pressure : overwrite[array[float]]
        Current absolute nodal pressure on the P1 pressure space.
    vorticity : overwrite[array[float]]
        Current scalar vorticity field, omega = dv/dx - du/dy (the 2D curl
        of velocity), interpolated onto the same P1 pressure space (so
        ``pressure_coords``/vorticity share dof coordinates) -- the
        canonical way to visualize a von Karman street: alternating
        positive/negative vorticity lobes shed off the cylinder.
    drag_coeff, lift_coeff : overwrite[float]
        Cd = 2*Fx/(rho*U_mean^2*D), Cl = 2*Fy/(rho*U_mean^2*D), where
        (Fx, Fy) is the real surface-force integral
        integral(sigma(u,p) . n) ds over the cylinder boundary facets
        (sigma = 2*mu*epsilon(u) - p*I, n the cylinder's own outward
        normal -- i.e. *minus* the fluid domain's facet normal, which
        points inward at this hole boundary). rho=1, D=2*cylinder_radius,
        and U_mean is the benchmark's *nominal* mean inflow velocity
        ((2/3)*u_peak, from integrating the parabolic profile) -- fixed
        regardless of ``inflow_ramp_time``, so Cd/Cl are directly
        comparable to the DFG reference values (Cd_max~=3.22-3.24,
        Cl_max~=0.99-1.01) once shedding is established.
    elapsed_time : overwrite[float]
        Total simulated time since this process instance was constructed
        (drives the inflow ramp; also useful for a caller building a
        Cd/Cl(t) time series without separately tracking tick count*dt).
    """

    config_schema = {
        "h_cylinder": {"_type": "float", "_default": 0.008},
        "h_far": {"_type": "float", "_default": 0.05},
        "reynolds": {"_type": "float", "_default": 100.0},
        "dt": {"_type": "float", "_default": 0.0005},
        "u_peak": {"_type": "float", "_default": 1.5},
        "inflow_ramp_time": {"_type": "float", "_default": 0.3},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._mesh_key = None
        self._domain = None
        self._facet_tags = None
        self._markers = None
        self._V = None
        self._Q = None
        self._bcu = None
        self._bcp = None
        self._u_inflow = None  # fem.Function(V), mutated in place each substep (ramp)
        self._u_n = None  # fem.Function(V), persisted velocity between ticks
        self._p_n = None  # fem.Function(Q), persisted pressure between ticks
        self._t = 0.0  # elapsed simulated time, drives the inflow ramp

    def _ensure_setup(self):
        h_cylinder = self.config["h_cylinder"]
        h_far = self.config["h_far"]
        mesh_key = (h_cylinder, h_far)
        if self._domain is None or self._mesh_key != mesh_key:
            self._domain, self._facet_tags, self._markers, self._V, self._Q = (
                _build_channel_spaces(h_cylinder, h_far)
            )
            self._bcu, self._bcp, self._u_inflow = _channel_cylinder_bcs(
                self._domain, self._facet_tags, self._markers, self._V, self._Q
            )
            self._u_n = fem.Function(self._V)
            self._p_n = fem.Function(self._Q)
            self._mesh_key = mesh_key
            self._t = 0.0

    def inputs(self):
        return {"inflow_perturbation": "float"}

    def outputs(self):
        return {
            "velocity_x": "overwrite[array[float]]",
            "velocity_y": "overwrite[array[float]]",
            "pressure": "overwrite[array[float]]",
            "vorticity": "overwrite[array[float]]",
            "drag_coeff": "overwrite[float]",
            "lift_coeff": "overwrite[float]",
            "elapsed_time": "overwrite[float]",
        }

    def initial_state(self):
        self._ensure_setup()
        return {"inflow_perturbation": 0.0}

    def _ipcs_channel_substep(self, u_peak_now, mu, dt):
        """One IPCS substep on the channel-cylinder mesh (see module-level
        comment above ``CylinderFlowProcess`` for the scheme; identical
        3-step structure to ``NavierStokesProcess._ipcs_substep``, but with
        the tagged channel BCs and a mutated-in-place inflow amplitude).
        """
        domain, V, Q = self._domain, self._V, self._Q
        bcu, bcp = self._bcu, self._bcp
        u_n, p_n = self._u_n, self._p_n
        gdim = domain.geometry.dim
        H = fem_gmsh.CHANNEL_HEIGHT

        self._u_inflow.interpolate(
            lambda x: np.stack(
                [4.0 * u_peak_now * x[1] * (H - x[1]) / H**2, np.zeros_like(x[1])]
            )
        )

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        p = ufl.TrialFunction(Q)
        q = ufl.TestFunction(Q)

        u_ = fem.Function(V)  # tentative velocity
        p_ = fem.Function(Q)  # corrected pressure

        k = fem.Constant(domain, PETSc.ScalarType(dt))
        mu_c = fem.Constant(domain, PETSc.ScalarType(mu))
        rho_c = fem.Constant(domain, PETSc.ScalarType(1.0))

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
            petsc_options_prefix=f"viva_fenics_cyl1_{next(_prefix_counter)}_",
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
            petsc_options_prefix=f"viva_fenics_cyl2_{next(_prefix_counter)}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        problem2.solve()

        # Step 3: velocity correction (solve directly into u_n).
        a3 = ufl.dot(u, v) * ufl.dx
        L3 = ufl.dot(u_, v) * ufl.dx - k * ufl.dot(ufl.grad(p_ - p_n), v) * ufl.dx
        problem3 = LinearProblem(
            a3, L3, bcs=[], u=u_n,
            petsc_options_prefix=f"viva_fenics_cyl3_{next(_prefix_counter)}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        problem3.solve()

        p_n.x.array[:] = p_.x.array

    def _drag_lift(self, mu, u_mean, diameter):
        """Real surface-force integral over the cylinder boundary -- see
        ``outputs`` docstring for the sign/normal convention.
        """
        domain = self._domain
        gdim = domain.geometry.dim
        n_fluid = ufl.FacetNormal(domain)
        n_obstacle = -n_fluid  # cylinder's own outward normal (see outputs docstring)
        ds_cyl = ufl.Measure(
            "ds", domain=domain, subdomain_data=self._facet_tags,
            subdomain_id=self._markers["cylinder"],
        )
        mu_c = fem.Constant(domain, PETSc.ScalarType(mu))

        def epsilon(uu):
            return ufl.sym(ufl.nabla_grad(uu))

        sigma_form = 2 * mu_c * epsilon(self._u_n) - self._p_n * ufl.Identity(gdim)
        traction = ufl.dot(sigma_form, n_obstacle)

        fx = domain.comm.allreduce(
            fem.assemble_scalar(fem.form(traction[0] * ds_cyl)), op=MPI.SUM
        )
        fy = domain.comm.allreduce(
            fem.assemble_scalar(fem.form(traction[1] * ds_cyl)), op=MPI.SUM
        )
        drag_coeff = 2.0 * fx / (1.0 * u_mean**2 * diameter)
        lift_coeff = 2.0 * fy / (1.0 * u_mean**2 * diameter)
        return float(drag_coeff), float(lift_coeff)

    def _vorticity(self):
        """Scalar vorticity omega = dv/dx - du/dy, interpolated onto the
        (P1) pressure space -- see ``outputs`` docstring.
        """
        u_n = self._u_n
        curl_2d = u_n[1].dx(0) - u_n[0].dx(1)
        expr = fem.Expression(curl_2d, self._Q.element.interpolation_points)
        omega = fem.Function(self._Q)
        omega.interpolate(expr)
        return omega.x.array.copy()

    def update(self, state, interval):
        self._ensure_setup()

        dt = self.config["dt"]
        reynolds = self.config["reynolds"]
        u_peak_full = self.config["u_peak"]
        ramp_time = self.config["inflow_ramp_time"]
        perturbation = float(state.get("inflow_perturbation") or 0.0)

        diameter = 2.0 * fem_gmsh.CYLINDER_RADIUS
        u_mean = (2.0 / 3.0) * u_peak_full  # mean of the parabolic inflow profile
        nu = u_mean * diameter / reynolds
        mu = 1.0 * nu

        n_steps = max(1, round(interval / dt))
        step_dt = interval / n_steps

        drag_coeff = lift_coeff = 0.0
        for _ in range(n_steps):
            self._t += step_dt
            ramp = min(1.0, self._t / ramp_time) if ramp_time > 0 else 1.0
            u_peak_now = ramp * u_peak_full + perturbation
            self._ipcs_channel_substep(u_peak_now, mu, step_dt)
            drag_coeff, lift_coeff = self._drag_lift(mu, u_mean, diameter)

        velocity_x = self._u_n.x.array[0::2].copy()
        velocity_y = self._u_n.x.array[1::2].copy()
        pressure = self._p_n.x.array.copy()
        vorticity = self._vorticity()

        return {
            "velocity_x": velocity_x,
            "velocity_y": velocity_y,
            "pressure": pressure,
            "vorticity": vorticity,
            "drag_coeff": drag_coeff,
            "lift_coeff": lift_coeff,
            "elapsed_time": self._t,
        }


def _build_channel_spaces(h_cylinder, h_far):
    """Build the DFG 2D-2 channel-cylinder mesh + Taylor-Hood-ish (P2
    velocity / P1 pressure) function space pair.

    Returns:
        (domain, facet_tags, markers, V, Q) tuple.
    """
    domain, facet_tags, markers = fem_gmsh.build_channel_cylinder_mesh(h_cylinder, h_far)
    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("Lagrange", 2, (gdim,)))
    Q = fem.functionspace(domain, ("Lagrange", 1))
    return domain, facet_tags, markers, V, Q


def _channel_cylinder_bcs(domain, facet_tags, markers, V, Q):
    """Dirichlet BCs for the DFG 2D-2 channel: parabolic inflow (velocity),
    no-slip on walls + cylinder (velocity), and zero pressure at the
    outflow (the standard "do-nothing" outflow condition -- velocity has no
    BC there, only a natural/traction-free condition, which the outflow
    pressure Dirichlet realizes weakly through the IPCS pressure-correction
    system).

    Returns:
        (bcu, bcp, u_inflow) tuple -- ``bcu``/``bcp`` are BC lists for the
        velocity/pressure systems; ``u_inflow`` is the (mutable) inflow
        ``fem.Function`` so the caller can update its amplitude in place
        each substep (see ``CylinderFlowProcess._ipcs_channel_substep``)
        without re-locating dofs or rebuilding the ``dirichletbc`` object.
    """
    fdim = domain.topology.dim - 1
    H = fem_gmsh.CHANNEL_HEIGHT

    def _facets(name):
        return facet_tags.find(markers[name])

    inflow_facets = _facets("inflow")
    outflow_facets = _facets("outflow")
    noslip_facets = np.concatenate([_facets("walls"), _facets("cylinder")])

    u_inflow = fem.Function(V)
    u_inflow.interpolate(lambda x: np.stack([np.zeros_like(x[1]), np.zeros_like(x[1])]))

    inflow_dofs = fem.locate_dofs_topological(V, fdim, inflow_facets)
    noslip_dofs = fem.locate_dofs_topological(V, fdim, noslip_facets)
    outflow_dofs = fem.locate_dofs_topological(Q, fdim, outflow_facets)

    bc_inflow = fem.dirichletbc(u_inflow, inflow_dofs)
    bc_noslip = fem.dirichletbc(PETSc.ScalarType((0.0, 0.0)), noslip_dofs, V)
    bc_outflow_p = fem.dirichletbc(PETSc.ScalarType(0.0), outflow_dofs, Q)

    return [bc_inflow, bc_noslip], [bc_outflow_p], u_inflow


def zero_channel_fields(h_cylinder=0.008, h_far=0.05):
    """Zero-valued (velocity_x, velocity_y, pressure, vorticity) nodal
    arrays of the correct length for a channel-cylinder mesh at this
    resolution, for seeding a composite's stores -- see ``zero_fields``
    (lid-driven cavity) for why never ``[]``.
    """
    _, _, _, V, Q = _build_channel_spaces(h_cylinder, h_far)
    n_v = V.tabulate_dof_coordinates().shape[0]
    n_q = Q.tabulate_dof_coordinates().shape[0]
    return (
        np.zeros(n_v, dtype=float),
        np.zeros(n_v, dtype=float),
        np.zeros(n_q, dtype=float),
        np.zeros(n_q, dtype=float),
    )


def channel_velocity_coords(h_cylinder=0.008, h_far=0.05):
    """(N, 2) dof coordinates of the P2 velocity space on the
    channel-cylinder mesh at this resolution."""
    _, _, _, V, _ = _build_channel_spaces(h_cylinder, h_far)
    return V.tabulate_dof_coordinates()[:, :2]


def channel_pressure_coords(h_cylinder=0.008, h_far=0.05):
    """(N, 2) dof coordinates of the P1 pressure space on the
    channel-cylinder mesh at this resolution -- also the vorticity field's
    coordinates (interpolated onto the same P1 space; see
    ``CylinderFlowProcess._vorticity``)."""
    _, _, _, _, Q = _build_channel_spaces(h_cylinder, h_far)
    return Q.tabulate_dof_coordinates()[:, :2]


# =============================================================================
# PorousFlowStep: steady Stokes flow through a periodic pillar lattice ->
# effective (Darcy) permeability
# =============================================================================
#
# A creeping-flow (Stokes, not Navier-Stokes) solve on the gmsh-generated
# porous-lattice mesh (``viva_fenics.fem_gmsh.build_porous_lattice_mesh``):
# a real Taylor-Hood (P2 velocity / P1 pressure) MIXED function space (one
# ``basix.ufl.mixed_element`` -- see the dolfinx 0.11 Stokes demo,
# ``demo_stokes.py``'s ``mixed_direct()``, for the API this follows), driven
# by a prescribed pressure DROP across the channel, with no-slip velocity on
# the channel walls AND every pillar. A SINGLE linear saddle-point
# ``LinearProblem`` solve (no time-stepping -- Stokes flow at vanishing
# Reynolds number has no unsteady term) produces the full velocity/pressure
# field, from which the volume-averaged x-velocity and Darcy's law give the
# effective permeability:
#
#     <u_x> = (k_eff / mu) * (delta_p / L)   =>   k_eff = mu * <u_x> * L / delta_p
#
# Pressure boundary condition (the "do-nothing"/Heywood-Rannacher-Turek
# pressure BC, standard for FEM Stokes/Navier-Stokes duct flow): velocity is
# left COMPLETELY UNCONSTRAINED at the inflow/outflow boundaries (no
# Dirichlet BC there at all -- only no-slip on walls+pillars), and the
# prescribed pressure enters as a NATURAL (Neumann) load on the momentum
# equation's right-hand side, ``-integral(p_prescribed * (n . v)) ds``,
# rather than as a Dirichlet constraint on the pressure trial space.
# Imposing p via a strong pressure-space Dirichlet BC INSTEAD (tried first,
# during development) is inconsistent with also dropping the momentum
# equation's own boundary integral (implicitly demanding zero TOTAL
# traction there) -- confirmed empirically: it drove ~zero net flow despite
# a nonzero pressure drop. The boundary-load form is the textbook-correct
# way to prescribe pressure while leaving the (near-parallel, viscous-
# stress-negligible) outflow physically free. Because velocity is
# Dirichlet-constrained on walls+pillars but left open at inflow/outflow,
# the mixed system has no pressure null space to remove (unlike the
# lid-driven-cavity Stokes demo, which is Dirichlet-velocity on its ENTIRE
# boundary and needs an explicit PETSc nullspace) -- a direct LU solve via
# the same ``LinearProblem`` convention used everywhere else in this module
# suffices.
POROUS_PRESSURE_DROP_DEFAULT = 1.0
POROUS_VISCOSITY_DEFAULT = 1.0


def _build_porous_spaces(nx, ny, pillar_radius, h_pillar, h_far):
    """Build the porous-lattice mesh + Taylor-Hood MIXED (P2 velocity / P1
    pressure) function space.

    Returns:
        (domain, facet_tags, markers, W, porosity) tuple -- ``W`` is the
        single mixed ``basix.ufl.mixed_element([P2, P1])`` function space
        (not a (V, Q) pair; the mixed/"non-blocked" formulation, per the
        dolfinx Stokes demo's ``mixed_direct()``).
    """
    domain, facet_tags, markers, porosity = fem_gmsh.build_porous_lattice_mesh(
        nx, ny, pillar_radius, h_pillar, h_far
    )
    gdim = domain.geometry.dim
    P2 = basix.ufl.element("Lagrange", domain.basix_cell(), 2, shape=(gdim,))
    P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    TH = basix.ufl.mixed_element([P2, P1])
    W = fem.functionspace(domain, TH)
    return domain, facet_tags, markers, W, porosity


def _porous_lattice_noslip_bc(domain, facet_tags, markers, W):
    """No-slip Dirichlet velocity BC on walls + every pillar (the ONLY
    Dirichlet BC in this problem -- see module comment above
    ``PorousFlowStep`` for why the pressure drop is imposed as a boundary
    LOAD, not a Dirichlet constraint).
    """
    fdim = domain.topology.dim - 1
    W0 = W.sub(0)
    V0, _ = W0.collapse()

    noslip_facets = np.concatenate([
        facet_tags.find(markers["walls"]), facet_tags.find(markers["pillars"]),
    ])
    noslip = fem.Function(V0)
    noslip.x.array[:] = 0.0
    noslip_dofs = fem.locate_dofs_topological((W0, V0), fdim, noslip_facets)
    return fem.dirichletbc(noslip, noslip_dofs, W0)


def solve_porous_stokes(nx, ny, pillar_radius, h_pillar, h_far, mu, pressure_drop):
    """Solve steady Stokes flow through the porous pillar lattice: one real
    mixed (Taylor-Hood) ``dolfinx.fem.petsc.LinearProblem`` saddle-point
    solve, no time-stepping (see module comment above ``PorousFlowStep``).

    Returns:
        (domain, facet_tags, markers, uh, ph, porosity) tuple -- ``uh``
        (velocity, P2 vector ``fem.Function``) and ``ph`` (pressure, P1
        scalar ``fem.Function``) are the COLLAPSED sub-functions of the
        mixed solution (independent function spaces/dof arrays, ready for
        the same nodal-array handling as ``NavierStokesProcess``/
        ``CylinderFlowProcess``).
    """
    domain, facet_tags, markers, W, porosity = _build_porous_spaces(
        nx, ny, pillar_radius, h_pillar, h_far
    )
    bc_noslip = _porous_lattice_noslip_bc(domain, facet_tags, markers, W)

    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)

    def epsilon(uu):
        return ufl.sym(ufl.grad(uu))

    mu_c = fem.Constant(domain, PETSc.ScalarType(mu))
    a = (
        2 * mu_c * ufl.inner(epsilon(u), epsilon(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        - q * ufl.div(u) * ufl.dx
    )

    # Pressure boundary LOAD (do-nothing pressure BC -- see module comment):
    # -integral(p_prescribed * (n . v)) ds at inflow (p = pressure_drop) and
    # outflow (p = 0), with the domain's own outward FacetNormal so the sign
    # comes out right at both ends without hand-flipping it per boundary.
    n = ufl.FacetNormal(domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    zero_force = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0)))
    p_in_c = fem.Constant(domain, PETSc.ScalarType(pressure_drop))
    p_out_c = fem.Constant(domain, PETSc.ScalarType(0.0))
    L = (
        ufl.inner(zero_force, v) * ufl.dx
        - p_in_c * ufl.inner(n, v) * ds(markers["inflow"])
        - p_out_c * ufl.inner(n, v) * ds(markers["outflow"])
    )

    prefix = f"viva_fenics_porous_{next(_prefix_counter)}_"
    problem = LinearProblem(
        a, L, bcs=[bc_noslip],
        petsc_options_prefix=prefix,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    wh = problem.solve()
    uh = wh.sub(0).collapse()
    ph = wh.sub(1).collapse()

    return domain, facet_tags, markers, uh, ph, porosity


def mean_velocity_x(domain, uh):
    """Volume-averaged x-velocity ``<u_x> = (1/Area) * integral(u_x) dx``
    over the fluid domain -- a real FEM integral (``fem.assemble_scalar``),
    not a nodal average, so it is exact for the P2 velocity field regardless
    of mesh node placement. The Darcy-law numerator (see module comment
    above ``PorousFlowStep``).
    """
    one = fem.Constant(domain, PETSc.ScalarType(1.0))
    area = float(domain.comm.allreduce(
        fem.assemble_scalar(fem.form(one * ufl.dx)), op=MPI.SUM
    ))
    ux_integral = float(domain.comm.allreduce(
        fem.assemble_scalar(fem.form(uh[0] * ufl.dx)), op=MPI.SUM
    ))
    return ux_integral / area


def noslip_max_speed(domain, facet_tags, markers, uh):
    """Max |u| over every no-slip (walls + pillars) boundary dof -- should
    sit at machine-precision zero if the Dirichlet BC was actually applied
    to the true tagged facets; a nonzero value here means the no-slip
    condition leaked (e.g. a pillar boundary misclassified into the wrong
    physical group).
    """
    fdim = domain.topology.dim - 1
    V = uh.function_space
    noslip_facets = np.concatenate([
        facet_tags.find(markers["walls"]), facet_tags.find(markers["pillars"]),
    ])
    noslip_dofs = fem.locate_dofs_topological(V, fdim, noslip_facets)
    if len(noslip_dofs) == 0:
        return 0.0
    vx = uh.x.array[0::2]
    vy = uh.x.array[1::2]
    speed = np.hypot(vx, vy)
    return float(np.max(speed[noslip_dofs]))


class PorousFlowStep(Step):
    """Steady Stokes (creeping) flow through a periodic circular-pillar
    lattice, driven by a prescribed pressure drop -- computes the effective
    (Darcy) permeability ``k_eff`` of the porous microstructure (see module
    comment above for the physics/BC setup).

    Stateless (a fresh mesh + a single linear saddle-point solve every
    invocation, no time-varying inputs) -- the Port-Design "no inputs =>
    Step" convention, same as ``PoissonSolverStep``.

    Outputs
    -------
    velocity_x, velocity_y : array[float]
        Nodal velocity components on the P2 velocity space (de-interleaved,
        same convention as ``NavierStokesProcess``).
    pressure : array[float]
        Nodal pressure on the P1 pressure space.
    porosity : float
        FEM-assembled fluid-area fraction of the imported mesh (see
        ``fem_gmsh.build_porous_lattice_mesh``).
    mean_ux : float
        Volume-averaged x-velocity ``<u_x>`` (Darcy-law numerator).
    k_eff : float
        Effective permeability, ``mu * mean_ux * channel_length /
        pressure_drop`` (Darcy's law, solved for k_eff).
    divergence_mean : float
        Mean |div(u)| over the domain -- should be small (a real Taylor-
        Hood/LBB-stable mixed solve is much closer to exactly
        divergence-free than IPCS operator splitting, which only achieves
        *approximate* incompressibility; see ``divergence_stats``).
    noslip_max_speed : float
        Max |u| over every no-slip (wall + pillar) boundary dof -- should be
        ~0 (see ``noslip_max_speed`` function docstring).
    n_cells : integer
        Imported mesh's local cell count (confirms a non-trivial
        triangulation, via ``fem_gmsh.n_cells``).
    """

    config_schema = {
        "nx": {"_type": "integer", "_default": 4},
        "ny": {"_type": "integer", "_default": 4},
        "pillar_radius": {"_type": "float", "_default": 0.08},
        "h_pillar": {"_type": "float", "_default": 0.015},
        "h_far": {"_type": "float", "_default": 0.05},
        "mu": {"_type": "float", "_default": POROUS_VISCOSITY_DEFAULT},
        "pressure_drop": {"_type": "float", "_default": POROUS_PRESSURE_DROP_DEFAULT},
    }

    def inputs(self):
        return {}

    def outputs(self):
        return {
            "velocity_x": "array[float]",
            "velocity_y": "array[float]",
            "pressure": "array[float]",
            "porosity": "float",
            "mean_ux": "float",
            "k_eff": "float",
            "divergence_mean": "float",
            "noslip_max_speed": "float",
            "n_cells": "integer",
        }

    def update(self, state):
        cfg = self.config
        domain, facet_tags, markers, uh, ph, porosity = solve_porous_stokes(
            cfg["nx"], cfg["ny"], cfg["pillar_radius"], cfg["h_pillar"], cfg["h_far"],
            cfg["mu"], cfg["pressure_drop"],
        )

        mean_ux = mean_velocity_x(domain, uh)
        k_eff = cfg["mu"] * mean_ux * fem_gmsh.POROUS_DOMAIN_LENGTH / cfg["pressure_drop"]
        divergence_mean, _divergence_max = divergence_stats(domain, uh.function_space, uh.x.array)
        max_noslip_speed = noslip_max_speed(domain, facet_tags, markers, uh)

        return {
            "velocity_x": uh.x.array[0::2].copy(),
            "velocity_y": uh.x.array[1::2].copy(),
            "pressure": ph.x.array.copy(),
            "porosity": porosity,
            "mean_ux": mean_ux,
            "k_eff": k_eff,
            "divergence_mean": divergence_mean,
            "noslip_max_speed": max_noslip_speed,
            "n_cells": fem_gmsh.n_cells(domain),
        }


def porous_velocity_coords(nx=4, ny=4, pillar_radius=0.08, h_pillar=0.015, h_far=0.05):
    """(N, 2) dof coordinates of the P2 velocity space on the porous-lattice
    mesh at this configuration."""
    _, _, _, W, _ = _build_porous_spaces(nx, ny, pillar_radius, h_pillar, h_far)
    V0, _ = W.sub(0).collapse()
    return V0.tabulate_dof_coordinates()[:, :2]


def porous_pressure_coords(nx=4, ny=4, pillar_radius=0.08, h_pillar=0.015, h_far=0.05):
    """(N, 2) dof coordinates of the P1 pressure space on the porous-lattice
    mesh at this configuration."""
    _, _, _, W, _ = _build_porous_spaces(nx, ny, pillar_radius, h_pillar, h_far)
    Q0, _ = W.sub(1).collapse()
    return Q0.tabulate_dof_coordinates()[:, :2]
