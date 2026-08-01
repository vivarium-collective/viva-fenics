"""Peristaltic pumping: a traveling occlusion wave on a channel's walls,
driving real incompressible flow via prescribed ALE (Arbitrary
Lagrangian-Eulerian) Navier-Stokes -- the investigation's flagship
composition showpiece, ``MovingBoundaryProcess``-style mesh motion
(``viva_fenics.processes.moving_boundary``) COMPOSED with
``NavierStokesProcess``-style IPCS flow (``viva_fenics.processes.flow``)
through shared bigraph stores, not implemented as one monolithic solve.

Physics
-------
A 2D channel ``[0, L] x [-H/2, H/2]``. The walls occlude as a traveling wave::

    occlusion(x, t) = (amplitude/2) * (1 + cos(2*pi*(x - wave_speed*t)/wavelength))

so the channel half-gap at ``x`` and time ``t`` is
``H/2 - occlusion(x, t)`` (symmetric: top wall moves to
``+H/2 - occlusion``, bottom wall mirrors to ``-H/2 + occlusion``) -- a
traveling constriction that squeezes fluid forward, the textbook peristalsis
mechanism.

Two processes, coupled ONLY through shared stores:

``PeristalticWallProcess`` (the mesh-motion half)
    Owns its OWN reference-configuration mesh (never deformed). Each tick,
    computes the prescribed wall displacement/velocity from the closed-form
    occlusion law above, then finds the INTERIOR mesh motion via a real
    harmonic (Laplacian) extension -- ``solve(-laplacian(phi) = 0)`` for a
    scalar vertical-displacement field ``phi``, Dirichlet-constrained to the
    prescribed value on the top/bottom wall facets and left natural
    (traction-free) at the channel ends -- the standard "Laplace smoothing"
    mesh-motion algorithm used in ALE mesh-motion tutorials. Solved TWICE per
    tick (once for total displacement, once for the time-derivative/mesh
    velocity, using the analytic time-derivative of the SAME boundary law as
    the Laplace BC -- valid because the Laplace operator is linear, so
    differentiating in time commutes with solving it). Outputs the resulting
    nodal fields as ``mesh_displacement_y`` / ``mesh_velocity_y``.

``PeristalticFlowProcess`` (the flow half)
    Owns its OWN flow mesh (Taylor-Hood-ish P2 velocity / P1 pressure, same
    IPCS operator-splitting scheme as ``NavierStokesProcess`` --
    see ``viva_fenics.processes.flow``'s module docstring for the 3-substep
    derivation), built with IDENTICAL geometry parameters (``L``, ``H``,
    ``nx``, ``ny``) as the wall process, so the two meshes' vertex layouts
    coincide 1:1 (see "geometry-order" note below). Each tick: (1) reads
    ``mesh_displacement_y`` and deforms ITS OWN mesh
    (``domain.geometry.x[:, 1] = reference_y + mesh_displacement_y`` --
    the same direct geometry-mutation technique
    ``MovingBoundaryProcess.deform_mesh`` uses), (2) reads
    ``mesh_velocity_y`` and builds a P1-vector mesh-velocity field
    ``u_mesh``, (3) solves ALE-Navier-Stokes: the classic 3-step IPCS
    scheme, but with the tentative-velocity step's convection term using the
    ALE-corrected convecting field ``u_n - u_mesh`` (not bare ``u_n``) --
    exactly the brief's ``(u - u_mesh).grad(u)`` correction -- and a no-slip
    velocity BC on the (now-moving) top/bottom walls equal to the LOCAL wall
    velocity (independently evaluated from this process's own copy of the
    occlusion law -- boundary DATA a process may legitimately derive from
    its own config, the same pattern ``CylinderFlowProcess`` uses for its
    inflow profile). No velocity BC at the channel ends -- only a zero-
    pressure "do-nothing" Dirichlet there (the brief's "inflow/outflow
    (zero-pressure)" option), so any net axial flow is driven PURELY by the
    wall motion, not an externally imposed pressure drop.

Geometry-order coupling
------------------------
The two processes are SEPARATE ``Process`` instances, each building its own
dolfinx mesh -- they do not share a live mesh object, only bigraph store
arrays. For that coupling to be numerically meaningful, both processes' P1
scalar/vector dof arrays must line up 1:1 with physical mesh points in an
order both sides agree on. Two facts (verified empirically for this
dolfinx/scipy install; ``_geometry_order_permutation`` computes an explicit
correspondence rather than silently assuming it, so a future dolfinx version
that reorders differently fails loudly instead of silently corrupting
physics):

1. ``dolfinx.mesh.create_rectangle`` is a deterministic, pure function of
   its arguments (comm size, corners, cell counts) -- two independently
   built meshes with identical arguments produce BIT-IDENTICAL
   ``domain.geometry.x`` layouts.
2. A P1 (scalar or vector) ``fem.functionspace``'s
   ``tabulate_dof_coordinates()`` may or may not match ``domain.geometry.x``
   ordering (dolfinx is free to reorder dofs for assembly performance).

So: every nodal array exchanged through a bigraph store (``mesh_displacement_y``,
``mesh_velocity_y``) is expressed in **geometry order** (``domain.geometry.x``'s
own ordering) -- guaranteed consistent between the two processes by fact (1) --
and each process independently converts to/from its own P1 space's native dof
order via a local ``_geometry_order_permutation`` (fact (2), computed once at
mesh-build time via nearest-point matching, with a hard assertion that the
match is exact to near machine precision).

Stability notes (from development)
-----------------------------------
An IMPULSIVE start (full occlusion amplitude at t=0) destabilizes the IPCS
solve for amplitude >~0.5 -- ``ramp_time`` linearly ramps the EFFECTIVE
amplitude from 0 over that many seconds (same fix
``CylinderFlowProcess.inflow_ramp_time`` uses for its own impulsive-start
instability); ``occlusion_and_rate`` computes the true analytic derivative
of the ramped law so the wall BC and the harmonic-extension mesh motion
never disagree even mid-ramp.

More fundamentally, diagnosed by directly inspecting cell areas of the
deforming mesh during development: pure-Laplacian ("harmonic smoothing")
mesh motion does not by itself guarantee a non-degenerate mesh under large
boundary displacement -- at ``amplitude=0.6`` (70% occlusion of this
module's default ``H=2.0``), the SMALLEST cell's area was observed to
shrink toward zero and, on a finer mesh, actually flip sign (a genuine
element inversion), which is what breaks the IPCS solve (not a CFL/timestep
issue -- reducing ``dt``, refining the mesh, or lowering the Reynolds number
did NOT prevent it; refining the mesh made the degeneracy appear at an
EARLIER, not later, occlusion fraction, since a finer mesh resolves the
local compression more precisely). A real fix (a stiffer mesh-motion
equation, e.g. linear elasticity, or local remeshing when quality degrades)
is future work; the honest, validated choice for THIS study is to keep the
occlusion amplitude sweep (<=0.4 at this module's default channel
dimensions/resolution) safely below the observed degeneracy onset
(empirically between amplitude~0.45-0.5), confirmed stable over multiple
wave periods at the production mesh (``nx=24``, ``ny=12``) -- see
``studies/moving-boundary/study.yaml``'s baseline/variants and
``sims/run.py``'s constants.
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
from scipy.spatial import cKDTree

from viva_fenics.processes.flow import mean_velocity_x

_prefix_counter = itertools.count()


# =============================================================================
# Shared geometry / physics helpers
# =============================================================================

def build_channel(L, H, nx, ny):
    """Build the (deterministic -- see module docstring) rectangle channel
    mesh ``[0, L] x [-H/2, H/2]``.
    """
    return mesh.create_rectangle(
        MPI.COMM_WORLD, [[0.0, -H / 2.0], [L, H / 2.0]], [int(nx), int(ny)],
        cell_type=mesh.CellType.triangle, diagonal=mesh.DiagonalType.crossed,
    )


def wave_phase(x, t, wavelength, wave_speed):
    return 2.0 * np.pi * (np.asarray(x, dtype=float) - wave_speed * t) / wavelength


def occlusion(x, t, amplitude, wavelength, wave_speed):
    """Inward wall displacement magnitude at (x, t) -- in [0, amplitude],
    the traveling-wave occlusion law (see module docstring).
    """
    theta = wave_phase(x, t, wavelength, wave_speed)
    return (amplitude / 2.0) * (1.0 + np.cos(theta))


def occlusion_rate(x, t, amplitude, wavelength, wave_speed):
    """Analytic d(occlusion)/dt at (x, t) -- closed form, not a finite
    difference (exact since occlusion(x, t) is smooth in t).
    """
    theta = wave_phase(x, t, wavelength, wave_speed)
    return amplitude * np.pi * wave_speed / wavelength * np.sin(theta)


def _ramp_value_and_deriv(t, ramp_time):
    """(ramp(t), d(ramp)/dt) for a linear 0->1 ramp over ``ramp_time``
    seconds, then constant 1 thereafter. ``ramp_time<=0`` disables the ramp
    (returns ``(1.0, 0.0)`` always).
    """
    if ramp_time is None or ramp_time <= 0:
        return 1.0, 0.0
    if t >= ramp_time:
        return 1.0, 0.0
    if t <= 0.0:
        return 0.0, 1.0 / ramp_time
    return t / ramp_time, 1.0 / ramp_time


def occlusion_and_rate(x, t, amplitude, wavelength, wave_speed, ramp_time=0.0):
    """(occlusion(x, t), d(occlusion)/dt) with the EFFECTIVE amplitude
    ramped linearly from 0 to ``amplitude`` over ``ramp_time`` seconds
    (``ramp_time<=0`` disables the ramp) -- i.e. the occlusion wave starts
    from a flat (undeformed) channel and smoothly grows to full amplitude,
    instead of snapping to full wall velocity at t=0. An impulsive
    (unramped) start was found to blow up the IPCS solve at amplitude>=0.6
    during development -- the wall's peak velocity scales with
    ``amplitude*pi*wave_speed/wavelength``, so a large amplitude means a
    large, discontinuous t=0 velocity jump; this is the same fix
    ``CylinderFlowProcess.inflow_ramp_time`` uses for its own
    impulsive-start instability.

    The rate is the TRUE analytic time-derivative of the ramped occlusion
    law (product rule: ``d(ramp*amp*shape)/dt = ramp'*amp*shape +
    ramp*amp*shape'``), not just ``occlusion_rate`` evaluated at the ramped
    amplitude -- both ``PeristalticWallProcess`` (mesh-motion Laplace BC)
    and ``PeristalticFlowProcess`` (wall no-slip velocity BC) call this SAME
    function so the prescribed wall velocity always exactly matches the
    wall's actual (ramped) displacement law, even during the ramp.
    """
    x = np.asarray(x, dtype=float)
    ramp, ramp_dot = _ramp_value_and_deriv(t, ramp_time)
    eff_amplitude = amplitude * ramp
    theta = wave_phase(x, t, wavelength, wave_speed)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    occ = (eff_amplitude / 2.0) * (1.0 + cos_t)
    occ_rate = (
        (amplitude * ramp_dot / 2.0) * (1.0 + cos_t)
        + eff_amplitude * np.pi * wave_speed / wavelength * sin_t
    )
    return occ, occ_rate


def _geometry_order_permutation(domain, space):
    """Permutation ``perm`` such that ``space.tabulate_dof_coordinates()[i]``
    sits at the SAME physical point as ``domain.geometry.x[perm[i]]`` -- see
    module docstring's "Geometry-order coupling" section. Verified via exact
    nearest-point matching (raises if any match isn't near machine
    precision, rather than silently trusting dof order == geometry order).
    """
    gdim = domain.geometry.dim
    space_coords = space.tabulate_dof_coordinates()[:, :gdim]
    geom_coords = domain.geometry.x[:, :gdim]
    tree = cKDTree(geom_coords)
    dist, perm = tree.query(space_coords)
    if dist.max() > 1e-8:
        raise RuntimeError(
            f"space<->geometry coordinate mismatch (max distance {dist.max():.3e}); "
            "expected an exact 1:1 point correspondence for a P1 space built "
            "directly on this mesh"
        )
    return perm


def to_geometry_order(perm, space_array):
    """Reorder a (space-native-order) nodal array into geometry order."""
    geom_array = np.empty_like(np.asarray(space_array))
    geom_array[perm] = space_array
    return geom_array


def from_geometry_order(perm, geom_array):
    """Reorder a geometry-order nodal array into (space-native) dof order."""
    return np.asarray(geom_array)[perm]


def zero_geometry_field(L, H, nx, ny):
    """Zero-valued array of length ``n_geometry_vertices`` for this channel
    configuration -- for seeding a composite's ``mesh_displacement_y`` /
    ``mesh_velocity_y`` stores (never ``[]`` -- see
    ``viva_fenics.composites.reaction_diffusion``'s docstring for why an
    empty starting array corrupts the first ``array[float]`` apply).
    """
    domain = build_channel(L, H, nx, ny)
    return np.zeros(domain.geometry.x.shape[0], dtype=float)


def zero_flow_fields(L, H, nx, ny):
    """Zero-valued (velocity_x, velocity_y, pressure) nodal arrays of the
    correct length for this channel's P2 velocity / P1 pressure spaces --
    for seeding a composite's stores (see ``zero_geometry_field`` for why
    never ``[]``; these are a DIFFERENT, larger length than the geometry
    field since P2 has extra edge-midpoint dofs beyond the mesh vertices).
    """
    domain = build_channel(L, H, nx, ny)
    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("Lagrange", 2, (gdim,)))
    Q = fem.functionspace(domain, ("Lagrange", 1))
    n_v = V.tabulate_dof_coordinates().shape[0]
    n_q = Q.tabulate_dof_coordinates().shape[0]
    return np.zeros(n_v, dtype=float), np.zeros(n_v, dtype=float), np.zeros(n_q, dtype=float)


def channel_velocity_coords(L, H, nx, ny):
    """(N, 2) REFERENCE (undeformed) dof coordinates of the P2 velocity
    space for this channel configuration -- for viz (the animation helper
    plots at fixed reference coordinates; see ``sims/run.py``'s viz
    docstring for the moving-mesh caveat this inherits from
    ``studies/moving-boundary``).
    """
    domain = build_channel(L, H, nx, ny)
    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("Lagrange", 2, (gdim,)))
    return V.tabulate_dof_coordinates()[:, :2]


# =============================================================================
# PeristalticWallProcess: harmonic-extension mesh motion
# =============================================================================

class PeristalticWallProcess(Process):
    """Prescribed traveling-wave wall motion + harmonic-extension interior
    mesh motion for the peristalsis composition (see module docstring).

    Inputs
    ------
    phase_drive : float
        Additive offset to elapsed time (default 0.0) -- a sibling process
        could perturb the wave's phase without a schema change, the same
        "real, if usually-zero, hook" convention
        ``MovingBoundaryProcess.boundary_drive`` uses.

    Outputs
    -------
    mesh_displacement_y, mesh_velocity_y : overwrite[array[float]]
        The harmonic-extension vertical displacement / velocity field, in
        GEOMETRY order (see module docstring) -- what
        ``PeristalticFlowProcess`` reads to deform its own mesh and build
        its ALE convective correction. ``overwrite``, not additive: a fresh
        absolute field every tick (same sensor convention as
        ``MovingBoundaryProcess.boundary_position``).
    wall_time : overwrite[float]
        This tick's elapsed simulated time (including ``phase_drive``) --
        ``PeristalticFlowProcess`` reads this (rather than tracking its own
        independent clock) to evaluate its wall no-slip velocity BC at
        EXACTLY the same instant the received ``mesh_displacement_y`` /
        ``mesh_velocity_y`` fields were computed at. Without this, the flow
        process's own clock and the (one-tick-lagged, per pbg's synchronous
        coupling model) mesh fields it reads drift out of sync as the
        wave's velocity grows during the amplitude ramp -- confirmed during
        development to blow up the IPCS solve at ~75% of a ramped
        amplitude=0.6, at every dt/mesh/Reynolds combination tried, until
        this single-clock fix was applied.
    wall_min_gap : overwrite[float]
        The current minimum full channel gap (``H - 2*max_x(occlusion(x,
        t))``) -- a real derived sensor (sampled from the same closed-form
        occlusion law, not a mesh-based reading) confirming the prescribed
        constriction is actually tightening/loosening as expected.
    """

    config_schema = {
        "L": {"_type": "float", "_default": 2.0},
        "H": {"_type": "float", "_default": 2.0},
        "nx": {"_type": "integer", "_default": 24},
        "ny": {"_type": "integer", "_default": 12},
        "amplitude": {"_type": "float", "_default": 0.3},
        "wavelength": {"_type": "float", "_default": 1.0},
        "wave_speed": {"_type": "float", "_default": 1.0},
        "ramp_time": {"_type": "float", "_default": 0.5},
        "dt": {"_type": "float", "_default": 0.01},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._domain = None
        self._Sy = None
        self._perm = None
        self._dof_x = None
        self._top_dofs = None
        self._bottom_dofs = None
        self._t = 0.0

    def _ensure_mesh(self):
        if self._domain is not None:
            return
        cfg = self.config
        self._domain = build_channel(cfg["L"], cfg["H"], cfg["nx"], cfg["ny"])
        self._Sy = fem.functionspace(self._domain, ("Lagrange", 1))
        self._perm = _geometry_order_permutation(self._domain, self._Sy)

        coords = self._Sy.tabulate_dof_coordinates()[:, :2]
        self._dof_x = coords[:, 0].copy()
        dof_y = coords[:, 1]
        H = cfg["H"]
        self._top_dofs = np.where(np.isclose(dof_y, H / 2.0))[0]
        self._bottom_dofs = np.where(np.isclose(dof_y, -H / 2.0))[0]
        if self._top_dofs.size == 0 or self._bottom_dofs.size == 0:
            raise RuntimeError("failed to locate top/bottom wall dofs on the reference channel mesh")

    def inputs(self):
        return {"phase_drive": "float"}

    def outputs(self):
        return {
            "mesh_displacement_y": "overwrite[array[float]]",
            "mesh_velocity_y": "overwrite[array[float]]",
            "wall_min_gap": "overwrite[float]",
            "wall_time": "overwrite[float]",
        }

    def initial_state(self):
        self._ensure_mesh()
        return {"phase_drive": 0.0}

    def _solve_harmonic(self, bc_top_vals, bc_bottom_vals):
        """One scalar Laplace solve for the vertical mesh-motion field with
        Dirichlet data ``bc_top_vals``/``bc_bottom_vals`` on the top/bottom
        wall dofs and natural (traction-free) conditions at the channel
        ends -- real dolfinx assemble+solve, fresh ``LinearProblem`` per
        call (same convention as every other solve in this codebase).
        """
        Sy = self._Sy
        phi = ufl.TrialFunction(Sy)
        psi = ufl.TestFunction(Sy)
        a = ufl.dot(ufl.grad(phi), ufl.grad(psi)) * ufl.dx
        zero = fem.Constant(self._domain, PETSc.ScalarType(0.0))
        L_form = zero * psi * ufl.dx

        bc_fn = fem.Function(Sy)
        bc_fn.x.array[self._top_dofs] = bc_top_vals
        bc_fn.x.array[self._bottom_dofs] = bc_bottom_vals
        all_dofs = np.concatenate([self._top_dofs, self._bottom_dofs])
        bc = fem.dirichletbc(bc_fn, all_dofs)

        prefix = f"viva_fenics_wallmotion_{next(_prefix_counter)}_"
        problem = LinearProblem(
            a, L_form, bcs=[bc],
            petsc_options_prefix=prefix,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        sol = problem.solve()
        return sol.x.array.copy()

    def update(self, state, interval):
        self._ensure_mesh()
        cfg = self.config
        phase_drive = float(state.get("phase_drive") or 0.0)
        amplitude, wavelength, wave_speed = cfg["amplitude"], cfg["wavelength"], cfg["wave_speed"]
        ramp_time = cfg["ramp_time"]

        dt = cfg["dt"]
        n_steps = max(1, round(interval / dt))
        step_dt = interval / n_steps

        top_x = self._dof_x[self._top_dofs]
        bottom_x = self._dof_x[self._bottom_dofs]

        disp_native = vel_native = None
        t_eff = self._t
        for _ in range(n_steps):
            self._t += step_dt
            t_eff = self._t + phase_drive

            occ_top, rate_top = occlusion_and_rate(top_x, t_eff, amplitude, wavelength, wave_speed, ramp_time)
            occ_bottom, rate_bottom = occlusion_and_rate(bottom_x, t_eff, amplitude, wavelength, wave_speed, ramp_time)
            disp_top, disp_bottom = -occ_top, occ_bottom
            rate_top, rate_bottom = -rate_top, rate_bottom

            disp_native = self._solve_harmonic(disp_top, disp_bottom)
            vel_native = self._solve_harmonic(rate_top, rate_bottom)

        disp_geom = to_geometry_order(self._perm, disp_native)
        vel_geom = to_geometry_order(self._perm, vel_native)

        x_samples = np.linspace(0.0, cfg["L"], 200)
        occ_samples, _ = occlusion_and_rate(x_samples, t_eff, amplitude, wavelength, wave_speed, ramp_time)
        wall_min_gap = cfg["H"] - 2.0 * float(occ_samples.max())

        return {
            "mesh_displacement_y": disp_geom,
            "mesh_velocity_y": vel_geom,
            "wall_min_gap": wall_min_gap,
            "wall_time": t_eff,
        }


# =============================================================================
# PeristalticFlowProcess: ALE incompressible Navier-Stokes on the deforming
# channel, driven by the wall process's mesh motion
# =============================================================================

class PeristalticFlowProcess(Process):
    """Real dolfinx ALE incompressible Navier-Stokes (IPCS operator
    splitting, ALE-corrected convection) on a channel deformed by
    ``PeristalticWallProcess``'s mesh motion (see module docstring).

    Inputs
    ------
    mesh_displacement_y, mesh_velocity_y : array[float]
        Geometry-order nodal fields written by ``PeristalticWallProcess``
        (falls back to a motionless mesh -- all zero -- if empty, e.g. the
        very first tick before the wall process has run).
    wall_time : float
        The simulated time ``PeristalticWallProcess`` used to compute the
        ABOVE two fields -- this process's own wall no-slip velocity BC is
        evaluated at this SAME time (not an independently-advanced internal
        clock), so the BC never gets out of sync with the mesh state it's
        applied to (see ``PeristalticWallProcess.outputs``' ``wall_time``
        docstring for why that matters).

    Outputs
    -------
    velocity_x, velocity_y : overwrite[array[float]]
        Current absolute nodal velocity components on the P2 velocity
        space (native dof order; see ``channel_velocity_coords`` for the
        matching REFERENCE coordinates). Overwrite, not additive -- same
        sensor convention as ``NavierStokesProcess.outputs``.
    pressure : overwrite[array[float]]
        Current absolute nodal pressure on the P1 pressure space.
    mean_ux : overwrite[float]
        Volume-averaged x-velocity ``<u_x>`` over the CURRENT (deformed,
        possibly area-varying) domain -- the per-tick net-axial-flow
        sensor; ``studies/moving-boundary/sims/run.py``'s (this study's)
        production run time-averages this over the run to report the net
        pumping rate Q.
    domain_area : overwrite[float]
        Real FEM-assembled current domain area (``integral(1) dx``) -- a
        sensor confirming the domain genuinely shrinks/grows with the
        traveling occlusion (unlike ``MovingBoundaryProcess``'s uniform
        stretch, peristaltic occlusion changes area non-uniformly in x).
    elapsed_time : overwrite[float]
        Total simulated time since this process instance was constructed.
    """

    config_schema = {
        "L": {"_type": "float", "_default": 2.0},
        "H": {"_type": "float", "_default": 2.0},
        "nx": {"_type": "integer", "_default": 24},
        "ny": {"_type": "integer", "_default": 12},
        "reynolds": {"_type": "float", "_default": 10.0},
        "amplitude": {"_type": "float", "_default": 0.3},
        "wavelength": {"_type": "float", "_default": 1.0},
        "wave_speed": {"_type": "float", "_default": 1.0},
        "ramp_time": {"_type": "float", "_default": 0.5},
        "dt": {"_type": "float", "_default": 0.01},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._domain = None
        self._V = None
        self._Q = None
        self._Um = None
        self._perm_um = None
        self._reference_y = None
        self._n_geom = None
        self._bcu = None
        self._bcp = None
        self._u_wall = None
        self._is_top_v = None
        self._is_bottom_v = None
        self._v_ref_x = None
        self._u_n = None
        self._p_n = None

    def _ensure_setup(self):
        if self._domain is not None:
            return
        cfg = self.config
        self._domain = build_channel(cfg["L"], cfg["H"], cfg["nx"], cfg["ny"])
        domain = self._domain
        gdim = domain.geometry.dim

        self._V = fem.functionspace(domain, ("Lagrange", 2, (gdim,)))
        self._Q = fem.functionspace(domain, ("Lagrange", 1))
        self._Um = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
        self._perm_um = _geometry_order_permutation(domain, self._Um)
        self._n_geom = domain.geometry.x.shape[0]

        # Captured BEFORE any deformation -- the fixed reference geometry
        # this process always deforms FROM (never incrementally, avoiding
        # drift; same convention as MovingBoundaryProcess.deform_mesh).
        self._reference_y = domain.geometry.x[:, 1].copy()

        fdim = domain.topology.dim - 1
        H = cfg["H"]

        def _top(x):
            return np.isclose(x[1], H / 2.0)

        def _bottom(x):
            return np.isclose(x[1], -H / 2.0)

        top_facets = mesh.locate_entities_boundary(domain, fdim, _top)
        bottom_facets = mesh.locate_entities_boundary(domain, fdim, _bottom)
        wall_facets = np.concatenate([top_facets, bottom_facets])

        end_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], cfg["L"]))
        )

        wall_dofs = fem.locate_dofs_topological(self._V, fdim, wall_facets)
        end_p_dofs = fem.locate_dofs_topological(self._Q, fdim, end_facets)

        self._u_wall = fem.Function(self._V)
        v_ref_coords = self._V.tabulate_dof_coordinates()[:, :2].copy()
        self._v_ref_x = v_ref_coords[:, 0]
        v_ref_y = v_ref_coords[:, 1]
        self._is_top_v = np.isclose(v_ref_y, H / 2.0)
        self._is_bottom_v = np.isclose(v_ref_y, -H / 2.0)

        bc_wall = fem.dirichletbc(self._u_wall, wall_dofs)
        bc_p_ends = fem.dirichletbc(PETSc.ScalarType(0.0), end_p_dofs, self._Q)
        self._bcu = [bc_wall]
        self._bcp = [bc_p_ends]

        self._u_n = fem.Function(self._V)
        self._p_n = fem.Function(self._Q)

    def inputs(self):
        return {
            "mesh_displacement_y": "array[float]",
            "mesh_velocity_y": "array[float]",
            "wall_time": "float",
        }

    def outputs(self):
        return {
            "velocity_x": "overwrite[array[float]]",
            "velocity_y": "overwrite[array[float]]",
            "pressure": "overwrite[array[float]]",
            "mean_ux": "overwrite[float]",
            "domain_area": "overwrite[float]",
            "elapsed_time": "overwrite[float]",
        }

    def initial_state(self):
        self._ensure_setup()
        return {
            "mesh_displacement_y": np.zeros(self._n_geom),
            "mesh_velocity_y": np.zeros(self._n_geom),
            "wall_time": 0.0,
        }

    def _update_wall_velocity_bc(self, t):
        cfg = self.config
        amplitude, wavelength, wave_speed = cfg["amplitude"], cfg["wavelength"], cfg["wave_speed"]
        ramp_time = cfg["ramp_time"]
        vy = np.zeros(self._v_ref_x.shape[0])
        _occ_top, rate_top = occlusion_and_rate(
            self._v_ref_x[self._is_top_v], t, amplitude, wavelength, wave_speed, ramp_time
        )
        _occ_bottom, rate_bottom = occlusion_and_rate(
            self._v_ref_x[self._is_bottom_v], t, amplitude, wavelength, wave_speed, ramp_time
        )
        vy[self._is_top_v] = -rate_top
        vy[self._is_bottom_v] = rate_bottom
        self._u_wall.x.array[0::2] = 0.0
        self._u_wall.x.array[1::2] = vy

    def _deform_mesh(self, mesh_displacement_y_geom):
        self._domain.geometry.x[:, 1] = self._reference_y + mesh_displacement_y_geom

    def _ale_ipcs_substep(self, u_mesh, mu, rho, dt):
        domain, V, Q = self._domain, self._V, self._Q
        u_n, p_n = self._u_n, self._p_n
        gdim = domain.geometry.dim

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        p = ufl.TrialFunction(Q)
        q = ufl.TestFunction(Q)

        u_ = fem.Function(V)
        p_ = fem.Function(Q)

        k = fem.Constant(domain, PETSc.ScalarType(dt))
        mu_c = fem.Constant(domain, PETSc.ScalarType(mu))
        rho_c = fem.Constant(domain, PETSc.ScalarType(rho))

        def epsilon(uu):
            return ufl.sym(ufl.nabla_grad(uu))

        def sigma(uu, pp):
            return 2 * mu_c * epsilon(uu) - pp * ufl.Identity(gdim)

        U_avg = 0.5 * (u_n + u)
        u_conv = u_n - u_mesh  # ALE convective correction: (u - u_mesh).grad(u)

        F1 = rho_c / k * ufl.dot(u - u_n, v) * ufl.dx
        F1 += rho_c * ufl.dot(ufl.dot(u_conv, ufl.nabla_grad(u_n)), v) * ufl.dx
        F1 += ufl.inner(sigma(U_avg, p_n), epsilon(v)) * ufl.dx
        a1 = ufl.lhs(F1)
        L1 = ufl.rhs(F1)
        problem1 = LinearProblem(
            a1, L1, bcs=self._bcu, u=u_,
            petsc_options_prefix=f"viva_fenics_peri1_{next(_prefix_counter)}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        problem1.solve()

        a2 = ufl.dot(ufl.grad(p), ufl.grad(q)) * ufl.dx
        L2 = (
            ufl.dot(ufl.grad(p_n), ufl.grad(q)) * ufl.dx
            - (rho_c / k) * ufl.div(u_) * q * ufl.dx
        )
        problem2 = LinearProblem(
            a2, L2, bcs=self._bcp, u=p_,
            petsc_options_prefix=f"viva_fenics_peri2_{next(_prefix_counter)}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        problem2.solve()

        a3 = ufl.dot(u, v) * ufl.dx
        L3 = ufl.dot(u_, v) * ufl.dx - k * ufl.dot(ufl.grad(p_ - p_n), v) * ufl.dx
        problem3 = LinearProblem(
            a3, L3, bcs=[], u=u_n,
            petsc_options_prefix=f"viva_fenics_peri3_{next(_prefix_counter)}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        problem3.solve()

        p_n.x.array[:] = p_.x.array

    def update(self, state, interval):
        self._ensure_setup()
        cfg = self.config

        disp_geom = np.asarray(state.get("mesh_displacement_y"))
        if disp_geom.size != self._n_geom:
            disp_geom = np.zeros(self._n_geom)
        vel_geom = np.asarray(state.get("mesh_velocity_y"))
        if vel_geom.size != self._n_geom:
            vel_geom = np.zeros(self._n_geom)
        wall_time = float(state.get("wall_time") or 0.0)

        H, wave_speed, reynolds = cfg["H"], cfg["wave_speed"], cfg["reynolds"]
        rho = 1.0
        nu = wave_speed * H / reynolds
        mu = rho * nu

        # Exactly ONE physical step of size `interval` per tick -- both
        # processes in the `peristalsis` composite are wired with
        # interval=dt (see viva_fenics.composites.peristalsis), so there is
        # no separate inner substep loop here (unlike NavierStokesProcess):
        # every substep needs a FRESH mesh-motion reading from
        # PeristalticWallProcess, which only arrives once per tick.

        # Deform the mesh + build the mesh-velocity field from the incoming
        # (per-tick) wall-process reading -- see module docstring's
        # "Geometry-order coupling" note.
        self._deform_mesh(disp_geom)
        u_mesh = fem.Function(self._Um)
        u_mesh.x.array[0::2] = 0.0
        u_mesh.x.array[1::2] = from_geometry_order(self._perm_um, vel_geom)

        # Evaluate the wall no-slip BC at the SAME `wall_time` the mesh
        # fields above were computed at (not an independently-advanced
        # clock) -- see PeristalticWallProcess.outputs' "wall_time"
        # docstring for why that consistency matters.
        self._update_wall_velocity_bc(wall_time)
        self._ale_ipcs_substep(u_mesh, mu, rho, interval)

        velocity_x = self._u_n.x.array[0::2].copy()
        velocity_y = self._u_n.x.array[1::2].copy()
        pressure = self._p_n.x.array.copy()

        mean_ux = mean_velocity_x(self._domain, self._u_n)

        one = fem.Constant(self._domain, PETSc.ScalarType(1.0))
        domain_area = float(self._domain.comm.allreduce(
            fem.assemble_scalar(fem.form(one * ufl.dx)), op=MPI.SUM
        ))

        return {
            "velocity_x": velocity_x,
            "velocity_y": velocity_y,
            "pressure": pressure,
            "mean_ux": mean_ux,
            "domain_area": domain_area,
            "elapsed_time": wall_time,
        }
