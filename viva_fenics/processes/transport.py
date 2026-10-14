"""AdvectionDiffusionProcess: a process-bigraph Process wrapping the FEniCSx
backward-Euler **advection-diffusion** (+ optional thermophoretic/Soret drift)
solve in :func:`viva_fenics.fem.advection_diffusion_step`.

This is the high-fidelity spatial-transport solver for the viva-genesis
origins-of-life campaign (full modeling plan section 4.5): the same backward-
Euler reactive-transport form as :class:`DiffusionProcess`, with two physics
terms added behind optional config/input ports so every existing viva-fenics
study that uses ``DiffusionProcess`` is bit-for-bit unaffected and a
``velocity=None / soret_coeff=0`` ``AdvectionDiffusionProcess`` reproduces
``DiffusionProcess`` exactly:

* **Advection** -- a Darcy upflow carrying the solute up the vent / pore
  (``velocity_mode``: ``uniform`` / ``stream_function`` use the configured
  ``(velocity_x, velocity_y)``; ``porous_flow`` consumes a resolved nodal
  velocity field wired from :class:`PorousFlowStep`), SUPG-stabilised at high
  cell Peclet number.
* **Thermophoretic (Soret) drift** -- the species drifts down the solved
  temperature gradient at ``w_T = -D * soret_coeff * grad(T)`` (the cold-side
  accumulation of the Baaske thermal trap), with ``T`` read from the
  ``temperature`` input port (a sibling heat-field ``DiffusionProcess`` /
  ``AdvectionDiffusionProcess``).

Like :class:`DiffusionProcess`, ``update()`` returns the field **delta**
(new - incoming) on "solution" so a sibling reaction process writing the same
store composes additively, and the absolute-field "integral" sensor as an
``overwrite[float]``.
"""

from __future__ import annotations

import numpy as np

from viva_fenics import fem
from viva_fenics.processes.diffusion import DiffusionProcess, _gaussian_bump


class AdvectionDiffusionProcess(DiffusionProcess):
    """Transient advection-diffusion (+ optional Soret drift) over a
    unit-square mesh, backward-Euler dolfinx timestepping.

    Inputs
    ------
    solution : array[float]
        Current absolute nodal field (``u_n``); falls back to the persisted
        field / initial bump when empty (as ``DiffusionProcess``).
    source : array[float]
        Additive nodal reaction/source rate (multiplied by dt internally).
    velocity : array[float]
        Resolved nodal velocity field on the vector ``(degree)`` space
        (interleaved x/y), consumed only when ``velocity_mode == "porous_flow"``
        -- e.g. wired from :class:`PorousFlowStep`'s velocity output. Ignored
        (zero default) for the ``uniform`` / ``stream_function`` modes, which
        use the configured constant upflow.
    temperature : array[float]
        Nodal temperature field driving the Soret drift (only used when
        ``soret_coeff != 0``); zero default leaves drift off.

    Outputs
    -------
    solution : array[float]
        Nodal delta (new - incoming) -- additive-store convention.
    integral : overwrite[float]
        FEM integral of the new absolute field (mass-conservation sensor).

    Config
    ------
    velocity_mode : string
        ``uniform`` / ``stream_function`` -> use ``(velocity_x, velocity_y)``;
        ``porous_flow`` -> use the ``velocity`` input field.
    velocity_x, velocity_y : float
        Constant upflow components for the uniform / stream_function modes.
    soret_coeff : float
        Soret coefficient S_T (1/K); 0 disables thermophoretic drift.
    supg : boolean
        SUPG stabilisation of the advection term (default True).
    """

    contract = {
        "summary": (
            "Real dolfinx backward-Euler advection-diffusion with optional "
            "thermophoretic (Soret) drift: the high-fidelity spatial-transport "
            "solver (advection-diffusion-reaction over the vent/pore domain)."
        ),
        "inputs": {
            "solution": "current absolute nodal field (u_n)",
            "source": "additive nodal reaction/source rate (x dt internally)",
            "velocity": "resolved nodal velocity field (porous_flow mode)",
            "temperature": "nodal temperature field driving Soret drift",
        },
        "outputs": {
            "solution": "nodal field delta (new - incoming), additive",
            "integral": "FEM integral of the new absolute field (overwrite sensor)",
        },
        "assumptions": [
            "Backward-Euler in time; P(degree) Lagrange elements on the unit square.",
            "Advection term dt*(w.grad(u))*v, SUPG-stabilised at Pe>1.",
            "Soret drift w_T = -D*soret*grad(T) folded into the advecting velocity; "
            "divergence-free for a linear T (conserves the field integral).",
            "velocity=None & soret_coeff=0 reduces exactly to DiffusionProcess.",
        ],
        "references": ["baaske2007extreme", "mast2010thermal"],
    }

    config_schema = {
        **DiffusionProcess.config_schema,
        "velocity_mode": {"_type": "string", "_default": "uniform"},
        "velocity_x": {"_type": "float", "_default": 0.0},
        "velocity_y": {"_type": "float", "_default": 0.0},
        "soret_coeff": {"_type": "float", "_default": 0.0},
        "supg": {"_type": "boolean", "_default": True},
    }

    def inputs(self):
        return {
            "source": "array[float]",
            "solution": "array[float]",
            "velocity": "array[float]",
            "temperature": "array[float]",
        }

    def initial_state(self):
        base = super().initial_state()
        self._ensure_mesh()
        n = base["solution"].shape[0]
        base["velocity"] = np.zeros(2 * n)
        base["temperature"] = np.zeros(n)
        return base

    def _resolve_velocity(self, state, n_dofs):
        mode = self.config["velocity_mode"]
        if mode == "porous_flow":
            vel = np.asarray(state.get("velocity"), dtype=float)
            if vel.size == 2 * n_dofs:
                return vel
            return None  # not yet wired; pure diffusion this tick
        wx, wy = self.config["velocity_x"], self.config["velocity_y"]
        if wx == 0.0 and wy == 0.0:
            return None
        return (wx, wy)

    def update(self, state, interval):
        self._ensure_mesh()

        incoming = np.asarray(state.get("solution"))
        if incoming.size == 0:
            prev = self._u_n if self._u_n is not None else self.initial_state()["solution"]
        else:
            prev = incoming.astype(float)

        source_in = np.asarray(state.get("source"))
        source = source_in.astype(float) if source_in.size == prev.size else np.zeros_like(prev)

        velocity = self._resolve_velocity(state, prev.size)

        soret = self.config["soret_coeff"]
        temperature = None
        if soret != 0.0:
            temp_in = np.asarray(state.get("temperature"))
            if temp_in.size == prev.size:
                temperature = temp_in.astype(float)

        dt = self.config["dt"]
        n_steps = max(1, round(interval / dt))
        step_dt = interval / n_steps
        bc_value = self.config["boundary_value"] if self.config["apply_boundary"] else None

        u = prev.copy()
        for _ in range(n_steps):
            u = fem.advection_diffusion_step(
                self._domain, self._V, u, source, step_dt, self.config["D"],
                velocity=velocity, soret=soret if temperature is not None else None,
                temperature_array=temperature, bc_value=bc_value,
                supg=self.config["supg"],
            )

        self._u_n = u.copy()
        integral = fem.field_integral(self._domain, self._V, u)
        return {"solution": u - prev, "integral": integral}
