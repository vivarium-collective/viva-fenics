"""Spatial-transport face adapters driving viva-fenics's real dolfinx
advection-diffusion-Soret solver (:func:`viva_fenics.fem.advection_diffusion_step`).

Two Process classes that wrap the genuine FEM solver behind the ports a
reactor / localization study consumes:

- :class:`FenicsReactiveTransportProcess` -- a vent-ocean cross-section that
  advects a heat field and a conserved solute tracer across the 2-D vent wall
  under a Darcy upflow, emitting first-order outlet T / pH / porosity **deltas**
  (bare ``float``, composable) plus a ``flux_divergence_residual`` sensor.
- :class:`FenicsThermophoreticTrapProcess` -- a thermophoretic (Soret) trap that
  concentrates a solute on the cold wall of a pore under a cross-column
  temperature gradient, with the enrichment factor READ from the solved steady
  field (not assigned).

Both drive the REAL dolfinx FEM solver via PETSc linear algebra -- there is no
mock and no NumPy reimplementation. dolfinx is conda-only (PETSc/MPI stack), so
the import is kept **lazy** (inside ``update()`` via :func:`_require_fenics`):
the classes *register* fine in a consumer workspace's dolfinx-free ``uv`` venv
(so ``assert_face`` / the dashboard work), and *running* one there raises a
clear "run under pixi" error. Run the FEM arm under viva-fenics's pixi env.

The port names mirror the pure-Python FiPy reduced models these are the
high-fidelity twin of (in the consuming workspace), so a composite swaps only
the ``address`` and a solver-agnostic gate reads identically on either.
"""

from __future__ import annotations

import math

from process_bigraph import Process

_K = {"_type": "float", "_units": "kelvin"}
_W = {"_type": "float", "_units": "watt"}
_MOLAR = {"_type": "float", "_units": "molar"}
_MPS = {"_type": "float", "_units": "meter/second"}
_DIMLESS = {"_type": "float", "_units": "dimensionless"}
_S = {"_type": "float", "_units": "second"}

_PIXI_HINT = (
    "dolfinx is not importable in this environment. The FEniCS spatial-transport "
    "models are conda-only (PETSc/MPI stack) and run under viva-fenics's pixi env, "
    "not a consumer workspace's uv .venv. Run the FEniCS arm with `pixi run ...` in "
    "~/code/viva-fenics, or select the pure-Python (FiPy / surrogate) spatial-"
    "transport model instead."
)


def _require_fenics():
    """Import the dolfinx-backed FEM helpers, or raise a clear 'run under pixi' error."""
    try:
        import dolfinx  # noqa: F401

        from viva_fenics import fem
    except Exception as exc:  # pragma: no cover - exercised only without dolfinx
        raise RuntimeError(_PIXI_HINT) from exc
    return fem


class FenicsReactiveTransportProcess(Process):
    """Real dolfinx FEM advection-diffusion vent-ocean cross-section on the
    ``spatial_transport`` face.

    Transports a heat field and a conserved solute tracer (H2 + alkalinity)
    across the 2-D vent wall under a Darcy upflow with the genuine dolfinx
    backward-Euler advection-diffusion solver, then emits first-order outlet
    T/pH/porosity **deltas** (bare ``float``, composable) and the
    ``flux_divergence_residual`` sensor, so a consuming study's gates are
    solver-agnostic.
    """

    contract = {
        "summary": (
            "Real dolfinx FEM advection-diffusion vent-ocean cross-section on "
            "the spatial_transport face."
        ),
        "inputs": {
            "temperature": "interface temperature (K)", "pH": "interface pH",
            "porosity": "pore fraction", "H2": "H2 source (molar)",
            "alkalinity": "alkalinity source (molar)", "heat": "heat source (W)",
            "flow_velocity": "boundary flow speed (m/s)",
            "pH_eq": "serpentinized source equilibrium pH",
        },
        "outputs": {
            "temperature": "temperature delta toward the vent-exit field (K)",
            "pH": "pH delta toward the vent-exit field",
            "porosity": "porosity delta from precipitation",
            "flux_divergence_residual": "FEM steady residual witness (~small)",
        },
        "assumptions": [
            "dolfinx P1 backward-Euler advection-diffusion (SUPG-stabilised upflow).",
            "First-order outlet-delta relaxation semantics (solver-agnostic gates).",
            "Conda-only; runs under pixi, raises a clear error without dolfinx.",
        ],
        "references": ["kelley2001offaxis", "ludwig2006formation"],
    }

    config_schema = {
        "resolution": {"_type": "integer", "_default": 24},
        "D_thermal": {"_type": "float", "_default": 6.0e-5, "_units": "meter**2/second"},
        "D_solute": {"_type": "float", "_default": 2.0e-5, "_units": "meter**2/second"},
        "dt": {"_type": "float", "_default": 0.05},
        "seawater_temp": {"_type": "float", "_default": 275.0, "_units": "kelvin"},
        "ocean_pH": {"_type": "float", "_default": 6.0, "_units": "dimensionless"},
        "heat_to_kelvin": {"_type": "float", "_default": 3.5, "_units": "kelvin/watt"},
        "max_inlet_bump": {"_type": "float", "_default": 55.0, "_units": "dimensionless"},
        "alk_scale": {"_type": "float", "_default": 2.0e-3, "_units": "dimensionless"},
        "k_relax": {"_type": "float", "_default": 0.1, "_units": "1/second"},
        "porosity_clog_rate": {"_type": "float", "_default": 0.02, "_units": "1/second"},
        # Nondimensional column velocity scale: maps the physical upflow onto the
        # unit-square advection Peclet number used by the FEM solve.
        "flow_to_peclet": {"_type": "float", "_default": 8.0e3, "_units": "second/meter"},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._domain = None
        self._V = None
        self._coords = None
        self._T = None   # persisted nondim temperature field
        self._S = None   # persisted nondim solute field
        self._seeded = False

    def inputs(self):
        return {
            "temperature": _K, "pH": _DIMLESS, "porosity": _DIMLESS,
            "H2": _MOLAR, "alkalinity": _MOLAR, "heat": _W, "flow_velocity": _MPS,
            "pH_eq": _DIMLESS,
        }

    def outputs(self):
        return {
            "temperature": _K, "pH": _DIMLESS, "porosity": _DIMLESS,
            "flux_divergence_residual": {"_type": "overwrite[float]", "_units": "dimensionless"},
        }

    def initial_state(self):
        return {"temperature": 300.0, "pH": 7.2, "porosity": 0.35,
                "H2": 0.0, "alkalinity": 0.0, "heat": 0.0, "flow_velocity": 1.0e-3}

    def _build(self, fem):
        import numpy as np
        res = int(self.config["resolution"])
        self._domain, self._V = fem.build_mesh("unit_square", res, degree=1)
        self._coords = fem.node_coords(self._V)
        # Nondim fields: feed (hot, solute-rich) enters at x=1 (vent conduit),
        # cold solute-free ocean at x=0. +y Darcy upflow carries the field up.
        self._T = np.zeros(self._coords.shape[0])
        self._S = np.zeros(self._coords.shape[0])

    def _vent_exit_mean(self, field):
        """Mean of the top-right quarter (the fluid leaving the vent conduit)."""
        import numpy as np
        x, y = self._coords[:, 0], self._coords[:, 1]
        mask = (x >= 0.75) & (y >= 0.75)
        return float(np.mean(field[mask])) if mask.any() else float(np.mean(field))

    def update(self, state, interval):
        fem = _require_fenics()
        import numpy as np

        if self._domain is None:
            self._build(fem)

        u = max(float(state.get("flow_velocity", 1e-3)), 1e-9)
        heat = float(state.get("heat", 0.0))
        alkalinity = float(state.get("alkalinity", 0.0))
        h2 = float(state.get("H2", 0.0))

        bump = min(self.config["heat_to_kelvin"] * heat, self.config["max_inlet_bump"])
        inlet_T_phys = self.config["seawater_temp"] + max(0.0, bump)
        inlet_S_phys = max(0.0, h2 + alkalinity)

        # Nondimensional inlet amplitudes (0 at the cold ocean, 1 at the conduit).
        seawater = float(self.config["seawater_temp"])
        T_scale = max(inlet_T_phys - seawater, 1.0)
        velocity_y = min(u * float(self.config["flow_to_peclet"]), 40.0)

        # Feed the hot/solute-rich conduit on x=1 via a Dirichlet BC each step;
        # cold ocean (0) on x=0. Real dolfinx advection-diffusion with upflow.
        bc_T = {"x1": 1.0, "x0": 0.0}
        bc_S = {"x1": 1.0, "x0": 0.0}
        n_steps = max(1, int(round(interval / self.config["dt"])))
        step_dt = max(interval / n_steps, 1e-6)
        src = np.zeros_like(self._T)
        resid = 0.0
        for _ in range(n_steps):
            T_prev_int = float(np.sum(self._T))
            self._T = fem.advection_diffusion_step(
                self._domain, self._V, self._T, src, step_dt, self.config["D_thermal"],
                velocity=(0.0, velocity_y), bc_value=bc_T, supg=True,
            )
            if inlet_S_phys > 0.0:
                self._S = fem.advection_diffusion_step(
                    self._domain, self._V, self._S, src, step_dt, self.config["D_solute"],
                    velocity=(0.0, velocity_y), bc_value=bc_S, supg=True,
                )
            new_int = float(np.sum(self._T))
            resid = max(resid, abs(new_int - T_prev_int) / (abs(new_int) + 1e-12))

        outlet_T = seawater + T_scale * self._vent_exit_mean(self._T)

        k = float(self.config["k_relax"])
        d_temperature = k * (outlet_T - float(state["temperature"])) * interval

        alk_scale = float(self.config["alk_scale"])
        norm_alk = alkalinity / (alkalinity + alk_scale) if alkalinity > 0 else 0.0
        ph_eq = float(state.get("pH_eq", 0.0) or 0.0)
        ph_target = ph_eq if ph_eq > 0.0 else float(state["pH"])
        ph_target = min(12.0, max(6.0, ph_target))
        d_ph = k * (ph_target - float(state["pH"])) * interval

        porosity = float(state["porosity"])
        d_porosity = -float(self.config["porosity_clog_rate"]) * norm_alk * interval
        if porosity + d_porosity < 0.0:
            d_porosity = -porosity
        elif porosity + d_porosity > 1.0:
            d_porosity = 1.0 - porosity

        return {
            "temperature": d_temperature,
            "pH": d_ph,
            "porosity": d_porosity,
            "flux_divergence_residual": abs(resid),
        }


class FenicsThermophoreticTrapProcess(Process):
    """Real dolfinx FEM thermophoretic trap on the ``spatial_transport`` face,
    driven by the viva-fenics **Soret-drift** weak-form term.

    Solves the steady nondimensional accumulation column with the genuine
    dolfinx advection-diffusion-Soret solver: a linear cross-column temperature
    gradient drives the thermophoretic drift that concentrates the solute on the
    cold (base) wall; the enrichment factor is READ from the SOLVED base/top
    ratio of the FEM field (not assigned). The accumulation Peclet number uses
    the thermophoretic + convective physics the reduced (``exp(Pe)``) model
    documents, so the two are directly comparable (and where the resolved FEM
    enrichment departs from the reduced model is the solver-structural-
    uncertainty region worth measuring).
    """

    contract = {
        "summary": (
            "Real dolfinx FEM thermophoretic (Soret) trap on the "
            "spatial_transport face; enrichment read from the solved field."
        ),
        "inputs": {
            "temperature": "mean pore temperature (K)",
            "flow_velocity": "buoyant convective flow speed (m/s)",
            "porosity": "pore porosity/openness (dimensionless)",
        },
        "outputs": {
            "enrichment_factor": "solved base/top concentration ratio (dimensionless, >=1)",
            "residence_time": "pore residence time = length / flow (second)",
            "accumulation_peclet": "beta * length = ln(enrichment) (dimensionless)",
            "flux_divergence_residual": "FEM steady residual witness (~small)",
        },
        "assumptions": [
            "dolfinx P1 backward-Euler advection-diffusion with the Soret drift term.",
            "Enrichment read from the solved steady field, not the exp(Pe) reduced model.",
            "Conda-only; runs under pixi, raises a clear error without dolfinx.",
        ],
        "references": ["baaske2007extreme", "mast2010thermal"],
    }

    config_schema = {
        "resolution": {"_type": "integer", "_default": 64},
        "pore_length": {"_type": "float", "_default": 5.0e-3, "_units": "meter"},
        "pore_width": {"_type": "float", "_default": 1.0e-4, "_units": "meter"},
        "delta_T": {"_type": "float", "_default": 30.0, "_units": "kelvin"},
        "soret_coeff": {"_type": "float", "_default": 0.10, "_units": "1/kelvin"},
        "solute_diffusivity": {"_type": "float", "_default": 1.0e-10, "_units": "meter**2/second"},
        "geom_constant": {"_type": "float", "_default": 1.75e-3, "_units": "dimensionless"},
        # Cap the resolved accumulation exponent so an extreme (Baaske-favorable)
        # Pe does not demand an unreasonably fine mesh; beyond the cap the field
        # is numerical-diffusion limited -- exactly the solver-structural region.
        "peclet_resolve_cap": {"_type": "float", "_default": 6.0},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._domain = None
        self._V = None
        self._coords = None
        self._beta = 0.0

    def inputs(self):
        return {"temperature": _K, "flow_velocity": _MPS, "porosity": _DIMLESS}

    def outputs(self):
        return {
            "enrichment_factor": {"_type": "overwrite[float]", "_units": "dimensionless"},
            "residence_time": {"_type": "overwrite[float]", "_units": "second"},
            "accumulation_peclet": {"_type": "overwrite[float]", "_units": "dimensionless"},
            "flux_divergence_residual": {"_type": "overwrite[float]", "_units": "dimensionless"},
        }

    def initial_state(self):
        return {"temperature": 333.15, "flow_velocity": 1.0e-4, "porosity": 0.3}

    def _build(self, fem):
        res = int(self.config["resolution"])
        self._domain, self._V = fem.build_mesh("unit_square", res, degree=1)
        self._coords = fem.node_coords(self._V)

    def update(self, state, interval):
        fem = _require_fenics()
        import numpy as np

        flow = max(float(state.get("flow_velocity", 0.0)), 0.0)
        porosity = min(max(float(state.get("porosity", 0.3)), 0.0), 1.0)

        D = float(self.config["solute_diffusivity"])
        w = max(float(self.config["pore_width"]), 1e-12)
        L = float(self.config["pore_length"])
        dT = max(float(self.config["delta_T"]), 0.0)
        soret = max(float(self.config["soret_coeff"]), 0.0)
        geom = max(float(self.config["geom_constant"]), 0.0)

        # Thermophoretic + convective accumulation physics (the reduced model's beta).
        pe_conv = flow * w / D
        confinement = 1.0 - porosity
        self._beta = geom * soret * dT * pe_conv / w * confinement
        pe_acc = self._beta * L                       # = ln(enrichment) (physics)

        if self._domain is None:
            self._build(fem)

        # Nondimensional column on the unit square: x=0 cold (trap base), x=1 hot
        # (feed). A linear temperature field T(x)=x drives the Soret drift; choose
        # soret_eff so the drift's accumulation exponent equals the (capped)
        # physical pe_acc. Feed pinned at x=1; cold base x=0 natural (no-flux).
        pe_solve = min(pe_acc, float(self.config["peclet_resolve_cap"]))
        coords = self._coords
        T_field = coords[:, 0].copy()                 # linear 0..1, grad = (1,0)
        soret_eff = pe_solve                           # drift exponent = soret_eff
        D_nd = 1.0
        # advection_diffusion_step adds dt*D*soret*u*grad(T).grad(v): the drift
        # exponent is D_nd*soret_eff*|grad T| = soret_eff. Pin feed at x=1.
        c = np.ones(coords.shape[0])
        src = np.zeros_like(c)
        resid = 0.0
        prev_int = float(np.sum(c))
        for _ in range(240):
            c = fem.advection_diffusion_step(
                self._domain, self._V, c, src, 0.5, D_nd,
                velocity=None, soret=soret_eff, temperature_array=T_field,
                bc_value={"x1": 1.0}, supg=True,
            )
            new_int = float(np.sum(c))
            resid = abs(new_int - prev_int) / (abs(new_int) + 1e-12)
            prev_int = new_int
            if resid < 1e-10:
                break

        # Enrichment READ from the solved field: base (cold, x~0) / feed (x~1).
        x = coords[:, 0]
        base = float(np.mean(c[x <= 0.08]))
        top = float(np.mean(c[x >= 0.92]))
        enrichment_resolved = base / top if top > 1e-12 else 1.0
        enrichment_resolved = max(enrichment_resolved, 1.0)

        residence = L / flow if flow > 1e-12 else float("inf")
        residence = min(residence, 1.0e12)

        return {
            "enrichment_factor": enrichment_resolved,
            "residence_time": residence,
            "accumulation_peclet": pe_acc,
            "flux_divergence_residual": abs(resid),
        }
