"""DiffusionProcess: a process-bigraph Process wrapping the FEniCSx
transient (backward-Euler) heat/diffusion equation in viva_fenics.fem.

This is a Process (not a Step) because it is time-varying and stateful --
each call to update() advances the diffusion field by `interval`, carrying
the previous nodal solution forward. Per the additive-store convention used
throughout pbg (see pbg_simbio.processes.SimbioProcess), update() returns a
**delta** (new field - previous field), not the absolute field, so that a
sibling process writing the same "solution" store composes additively.
"""

from __future__ import annotations

import numpy as np
from process_bigraph import Process

from viva_fenics import fem

# No Dirichlet boundary condition is applied in fem.diffusion_step, so the
# domain boundary is natural Neumann (zero-flux): heat/mass cannot leave
# through the boundary, only redistribute -- hence the mass-conservation
# check in the test.


def _gaussian_bump(coords, center=(0.5, 0.5), sigma=0.1, amplitude=1.0):
    """Nodal values of a centered 2D gaussian bump over mesh dof coords."""
    dx = coords[:, 0] - center[0]
    dy = coords[:, 1] - center[1]
    return amplitude * np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))


class DiffusionProcess(Process):
    """Transient diffusion (heat equation) over a unit-square mesh, solved
    with backward-Euler dolfinx timestepping.

    Inputs
    ------
    solution : array[float]
        Current absolute nodal field (used as u_n, the previous timestep's
        field, for this update). Falls back to the process's own persisted
        field (or its initial gaussian bump) if empty.
    source : array[float]
        Additive nodal source term applied each substep (a sibling process,
        e.g. a reaction term, can write into the same store).

    Outputs
    -------
    solution : array[float]
        Nodal delta (new field - incoming field) so the shared array[float]
        store accumulates additively.
    integral : float
        FEM integral of the new (absolute) field, for mass-conservation
        bookkeeping.

    Config
    ------
    apply_boundary / boundary_value : bool / float
        Optional real Dirichlet BC pinning the x=0 face to `boundary_value`
        every step (e.g. a morphogen source boundary c=c0) -- see
        `viva_fenics.composites.morphogen_gradient`. Off by default
        (`apply_boundary=False`), so this is a strict opt-in extension: every
        pre-existing composite that doesn't set it gets exactly the original
        all-Neumann behavior.
    """

    config_schema = {
        "resolution": {"_type": "integer", "_default": 32},
        "degree": {"_type": "integer", "_default": 1},
        "D": {"_type": "float", "_default": 0.1},
        "dt": {"_type": "float", "_default": 0.01},
        "initial": {"_type": "string", "_default": "gaussian"},
        # A morphogen-gradient-style source boundary: when `apply_boundary`
        # is True, the x=0 face is pinned to a real Dirichlet BC at
        # `boundary_value` (e.g. c=c0) every step, via fem.diffusion_step's
        # `bc_value`; every other face stays natural Neumann (zero-flux).
        # Off by default so every existing composite (transient_diffusion,
        # reaction_diffusion, turing_patterns) is bit-for-bit unaffected.
        "apply_boundary": {"_type": "boolean", "_default": False},
        "boundary_value": {"_type": "float", "_default": 0.0},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._resolution = None
        self._domain = None
        self._V = None
        self._u_n = None  # persisted previous nodal field between calls

    def _ensure_mesh(self):
        resolution = self.config["resolution"]
        if self._domain is None or self._resolution != resolution:
            self._domain, self._V = fem.build_mesh(
                "unit_square", resolution, degree=self.config["degree"]
            )
            self._resolution = resolution

    def inputs(self):
        return {"source": "array[float]", "solution": "array[float]"}

    def outputs(self):
        # "integral" is a sensor reading (the FEM integral of the CURRENT
        # absolute field, recomputed fresh every tick in update()), not a
        # delta -- unlike "solution" it must REPLACE the store, not add to
        # it, or a multi-tick Composite run would accumulate
        # sum_i integral_i (~= N x mass) instead of reporting the true
        # per-tick reading. `overwrite[float]` gives it apply=replace
        # semantics (same convention as e.g. pbg_idynomics2's "time" output).
        return {"solution": "array[float]", "integral": "overwrite[float]"}

    def initial_state(self):
        self._ensure_mesh()
        coords = fem.node_coords(self._V)
        if self.config["initial"] == "gaussian":
            bump = _gaussian_bump(coords)
        elif self.config["initial"] == "zero":
            bump = np.zeros(coords.shape[0])
        else:
            raise ValueError(f"Unsupported initial condition: {self.config['initial']!r}")
        self._u_n = bump.copy()
        return {
            "solution": bump,
            "source": np.zeros_like(bump),
        }

    def update(self, state, interval):
        self._ensure_mesh()

        incoming = np.asarray(state.get("solution"))
        if incoming.size == 0:
            prev = self._u_n if self._u_n is not None else self.initial_state()["solution"]
        else:
            prev = incoming.astype(float)

        source_in = np.asarray(state.get("source"))
        source = source_in.astype(float) if source_in.size == prev.size else np.zeros_like(prev)

        dt = self.config["dt"]
        n_steps = max(1, round(interval / dt))
        step_dt = interval / n_steps

        bc_value = self.config["boundary_value"] if self.config["apply_boundary"] else None

        u = prev.copy()
        for _ in range(n_steps):
            u = fem.diffusion_step(
                self._domain, self._V, u, source, step_dt, self.config["D"], bc_value=bc_value,
            )

        self._u_n = u.copy()
        integral = fem.field_integral(self._domain, self._V, u)

        return {
            "solution": u - prev,
            "integral": integral,
        }
