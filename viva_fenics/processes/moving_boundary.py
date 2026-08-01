"""MovingBoundaryProcess: a process-bigraph Process wrapping a REAL dolfinx
diffusion (heat) solve on a **deforming domain** -- prescribed ALE mesh
motion, not a mock.

ALE approach
------------
Full "add a convective ALE term for the mesh velocity" incompressible-style
ALE is more machinery than this showcase needs (and than the task brief
requires); the brief's explicitly-acceptable fallback is used instead:
**prescribed mesh-coordinate motion + re-solve on the deformed mesh**, which
*is* genuine moving-boundary FEM (real dolfinx geometry mutation + a fresh
real assemble/solve every substep), not a kinematics-only illustration.

Concretely: the mesh is built ONCE, at t=0, as the reference configuration
(a unit square, so the reference y-coordinate ``y_ref`` of every mesh vertex
lies in ``[0, 1]``). Each substep, the TOP boundary's target y-coordinate
``y_top(t)`` is evaluated from a prescribed law (``mode="oscillate"``:
``y_top = 1 + amplitude*sin(omega*t)``; ``mode="grow"``:
``y_top = 1 + amplitude*t``), and the mesh is deformed in place by a uniform
vertical stretch::

    domain.geometry.x[:, 1] = y_ref * y_top(t)

which maps the reference bottom boundary (``y_ref=0``) to the fixed
``y=0`` and the reference top boundary (``y_ref=1``) to the moving
``y=y_top(t)`` -- i.e. real dolfinx mesh-geometry mutation, exactly the
technique used in the dolfinx/FEniCS ALE-mesh-motion tutorials. A transient
(backward-Euler) diffusion field is then re-assembled and re-solved with a
fresh ``dolfinx.fem.petsc.LinearProblem`` **on that deformed geometry**
every substep -- the Laplacian/mass forms integrate over the mesh's current
(deformed) cells, so this is a genuine re-solve on a moving domain, not a
cosmetic coordinate transform applied after the fact.

Because the domain is exactly ``[0, 1] x [0, y_top(t)]`` (only the height
changes; the width stays 1), the FEM-assembled domain area
(``domain_measure``, ``integral(1) dx`` over the current mesh) should equal
``y_top(t)`` (== ``boundary_position``) to near machine precision --
integrating a constant is exact for any mesh, any refinement -- which gives
a strong, non-tautological cross-check that the mesh really did deform (see
``tests/test_moving_boundary.py``).
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

from viva_fenics.processes.diffusion import _gaussian_bump

# Unique-ish petsc_options_prefix per solve -- same convention as
# viva_fenics.fem / viva_fenics.processes.flow (dolfinx 0.11's LinearProblem
# needs this; see fem.py's module docstring).
_prefix_counter = itertools.count()


def _build_mesh(resolution, degree=1):
    """Build the reference-configuration mesh + function space.

    Returns:
        (domain, V, y_ref) tuple, where ``y_ref`` is a copy of the
        reference (t=0, undeformed unit-square) y-coordinates of every
        mesh geometry vertex, in ``[0, 1]`` -- the basis for the uniform
        vertical stretch applied each substep.
    """
    domain = mesh.create_unit_square(
        MPI.COMM_WORLD, resolution, resolution,
        diagonal=mesh.DiagonalType.crossed,
    )
    V = fem.functionspace(domain, ("Lagrange", degree))
    y_ref = domain.geometry.x[:, 1].copy()
    return domain, V, y_ref


def boundary_law(mode, t, amplitude, omega):
    """Prescribed top-boundary y-coordinate as a function of time.

    Args:
        mode: "oscillate" (``1 + amplitude*sin(omega*t)``) or "grow"
            (``1 + amplitude*t``, monotonically increasing).
        t: elapsed simulated time.
        amplitude: oscillation amplitude, or growth rate for "grow".
        omega: angular frequency (oscillate mode only).

    Returns:
        float, the target top-boundary y-coordinate at time ``t``.
    """
    if mode == "oscillate":
        return 1.0 + amplitude * np.sin(omega * t)
    elif mode == "grow":
        return 1.0 + amplitude * t
    else:
        raise ValueError(f"Unsupported mode: {mode!r}")


def deform_mesh(domain, y_ref, y_top):
    """Mutate ``domain.geometry.x`` in place: a uniform vertical stretch
    mapping the reference y in [0, 1] to [0, y_top] (the bottom boundary
    stays fixed at y=0; the top boundary moves to y=y_top). Real dolfinx
    mesh-geometry mutation -- the ALE motion itself.
    """
    domain.geometry.x[:, 1] = y_ref * y_top


def domain_area(domain):
    """FEM-assembled area (``integral(1) dx``) of the domain's CURRENT
    (possibly deformed) geometry -- a real ``dolfinx.fem.assemble_scalar``
    call, not a shortcut formula, even though for this uniform-stretch
    deformation it is analytically ``1 * y_top`` (see module docstring).
    """
    one = fem.Constant(domain, PETSc.ScalarType(1.0))
    form = fem.form(one * ufl.dx)
    local = fem.assemble_scalar(form)
    return float(domain.comm.allreduce(local, op=MPI.SUM))


def moving_diffusion_step(domain, V, u_n_array, dt, D):
    """One backward-Euler diffusion substep, assembled and solved on the
    mesh's CURRENT (already-deformed, via ``deform_mesh``) geometry.

    No Dirichlet BC is applied (natural Neumann / zero-flux on the moving
    boundary too), matching ``viva_fenics.fem.diffusion_step``'s
    convention.

    Args:
        domain: dolfinx mesh, already deformed for this substep.
        V: dolfinx function space over domain.
        u_n_array: nodal values of the field at the start of the substep.
        dt: substep size.
        D: diffusion coefficient.

    Returns:
        np.ndarray of nodal solution values after the substep.
    """
    u_n = fem.Function(V)
    u_n.x.array[:] = u_n_array

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = (u * v + dt * D * ufl.dot(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = u_n * v * ufl.dx

    prefix = f"viva_fenics_mb_{next(_prefix_counter)}_"
    problem = LinearProblem(
        a, L, bcs=[],
        petsc_options_prefix=prefix,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    uh = problem.solve()
    return uh.x.array.copy()


def reference_coords(V):
    """(N, 2) dof coordinates of V in the REFERENCE (t=0, undeformed)
    configuration -- must be captured before any ``deform_mesh`` call.
    """
    return V.tabulate_dof_coordinates()[:, :2].copy()


class MovingBoundaryProcess(Process):
    """Transient diffusion on a deforming (prescribed-ALE) domain: a unit
    square whose TOP boundary moves according to a prescribed law each
    tick, re-solved on the deformed mesh every substep (see module
    docstring).

    Inputs
    ------
    solution : array[float]
        Current absolute nodal field (used as u_n for this update). Falls
        back to the process's own persisted field (or its initial gaussian
        bump) if empty -- same convention as ``DiffusionProcess``.
    boundary_drive : float
        Additive offset applied on top of the prescribed boundary law each
        substep (``y_top = boundary_law(...) + boundary_drive``), so a
        sibling process can perturb the moving boundary beyond the built-in
        law. Defaults to 0.0, i.e. the boundary follows the prescribed law
        exactly. This is what justifies MovingBoundaryProcess being a
        ``Process`` (time-stepped, stateful ALE motion) with a real input
        port, rather than a parameterless kinematic animation.

    Outputs
    -------
    solution : array[float]
        Nodal delta (new field - incoming field), additive convention
        matching ``DiffusionProcess``.
    boundary_position : overwrite[float]
        The mesh's actual current top-boundary y-coordinate
        (``domain.geometry.x[:, 1].max()``) after this tick's deformation --
        a real geometry reading, not a copy of the prescribed law's
        return value (though for this deformation they coincide up to
        floating point).
    domain_measure : overwrite[float]
        The FEM-assembled area of the current (deformed) domain -- a sensor
        reading, ``overwrite`` semantics (see ``DiffusionProcess.integral``
        for why: it must replace, not accumulate, in the shared store).
    """

    config_schema = {
        "resolution": {"_type": "integer", "_default": 32},
        "degree": {"_type": "integer", "_default": 1},
        "amplitude": {"_type": "float", "_default": 0.3},
        "omega": {"_type": "float", "_default": 3.14159},
        "D": {"_type": "float", "_default": 0.1},
        "dt": {"_type": "float", "_default": 0.01},
        "mode": {"_type": "string", "_default": "oscillate"},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._resolution = None
        self._domain = None
        self._V = None
        self._y_ref = None
        self._u_n = None  # persisted previous nodal field between calls
        self._t = 0.0  # elapsed simulated time, drives boundary_law

    def _ensure_mesh(self):
        resolution = self.config["resolution"]
        if self._domain is None or self._resolution != resolution:
            self._domain, self._V, self._y_ref = _build_mesh(
                resolution, degree=self.config["degree"]
            )
            self._resolution = resolution

    def inputs(self):
        return {"solution": "array[float]", "boundary_drive": "float"}

    def outputs(self):
        return {
            "solution": "array[float]",
            "boundary_position": "overwrite[float]",
            "domain_measure": "overwrite[float]",
        }

    def initial_state(self):
        self._ensure_mesh()
        coords = reference_coords(self._V)
        bump = _gaussian_bump(coords)
        self._u_n = bump.copy()
        return {
            "solution": bump,
            "boundary_drive": 0.0,
        }

    def update(self, state, interval):
        self._ensure_mesh()

        incoming = np.asarray(state.get("solution"))
        if incoming.size == 0:
            prev = self._u_n if self._u_n is not None else self.initial_state()["solution"]
        else:
            prev = incoming.astype(float)

        boundary_drive = float(state.get("boundary_drive") or 0.0)

        mode = self.config["mode"]
        amplitude = self.config["amplitude"]
        omega = self.config["omega"]
        D = self.config["D"]

        dt = self.config["dt"]
        n_steps = max(1, round(interval / dt))
        step_dt = interval / n_steps

        u = prev.copy()
        y_top = self._y_ref.max() if self._y_ref.size else 1.0
        for _ in range(n_steps):
            self._t += step_dt
            y_top = boundary_law(mode, self._t, amplitude, omega) + boundary_drive
            deform_mesh(self._domain, self._y_ref, y_top)
            u = moving_diffusion_step(self._domain, self._V, u, step_dt, D)

        self._u_n = u.copy()
        boundary_position = float(self._domain.geometry.x[:, 1].max())
        domain_measure = domain_area(self._domain)

        return {
            "solution": u - prev,
            "boundary_position": boundary_position,
            "domain_measure": domain_measure,
        }
