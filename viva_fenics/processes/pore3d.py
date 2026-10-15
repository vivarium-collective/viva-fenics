"""MultisolutePoreStep: multisolute reactive transport in a representative 3-D
pore, wrapped as a process-bigraph Step.

Like ``AdaptiveRefinementStep`` this is a Step: given its config it
deterministically runs the WHOLE pore simulation (build the 3-D
converging-diverging pore -> integrate every solute to near steady state ->
read the per-species enrichment + the mixture's co-localization) rather than
modelling one time tick. It is the high-fidelity FEM reference the full
modeling plan's Stage-III co-localization test (§4.5) is measured against.

``species`` is a ``tree`` (a list of per-solute dicts). The Step default is a
small, deliberately **illustrative, nondimensional** spread of diffusivities
and Soret coefficients -- NOT literature values -- so the composite is
runnable out of the box; the real 8-solute prebiotic mixture (phosphate,
ammonium, formate, acetate, pyruvate, amino-acid and nucleotide precursors,
amphiphiles) with sourced ``D``/``S_T``/adsorption/degradation is supplied by
the study's parameter package in viva-genesis.

Requires dolfinx (conda-only); runs under viva-fenics's pixi env.
"""
from __future__ import annotations

import itertools

import numpy as np
from process_bigraph import Step

# Illustrative, nondimensional default mixture (NOT literature values): three
# solutes spanning a range of diffusivity and thermophoretic strength so the
# default run exhibits species-specific fractionation.
_DEFAULT_SPECIES = [
    {"name": "fast_weak", "D": 0.08, "soret": 0.04, "inlet": 1.0},
    {"name": "mid", "D": 0.05, "soret": 0.09, "inlet": 1.0},
    {"name": "slow_strong", "D": 0.03, "soret": 0.15, "inlet": 1.0},
]


class MultisolutePoreStep(Step):
    """Run a multisolute Soret + adsorption + degradation transport simulation
    in a representative 3-D converging-diverging pore and report, per species,
    the trap enrichment (peak/feed) and, for the mixture, the worst-pair spatial
    co-localization correlation (``colocalization_min``: ``+1`` the whole
    mixture concentrates together, low/negative a fractionating pair -- the
    incompatibility the plan asks to be reported)."""

    config_schema = {
        "length": {"_type": "float", "_default": 1.0},
        "chamber_radius": {"_type": "float", "_default": 0.4},
        "throat_radius": {"_type": "float", "_default": 0.12},
        "h_throat": {"_type": "float", "_default": 0.04},
        "h_chamber": {"_type": "float", "_default": 0.10},
        "peclet": {"_type": "float", "_default": 8.0},
        "delta_T": {"_type": "float", "_default": 30.0},
        "dt": {"_type": "float", "_default": 0.02},
        "n_steps": {"_type": "integer", "_default": 120},
        "species": {"_type": "tree", "_default": _DEFAULT_SPECIES},
    }

    def inputs(self):
        return {}

    def outputs(self):
        # enrichment/peak are per-species in the config's species order (the
        # names are config-known, so they are not re-emitted as a tree port --
        # that keeps every emitted port a concrete, resolvable schema type).
        return {
            "enrichment": "array[float]",
            "peak": "array[float]",
            "colocalization_min": "float",
            "colocalization_mean": "float",
            "n_species": "integer",
        }

    def update(self, state):
        # Imported here so the module registers in the uv .venv (no dolfinx) and
        # only *runs* under pixi -- same lazy-import contract as the rest of the
        # FEM path.
        from dolfinx import fem

        from viva_fenics.fem_multisolute import (
            colocalization,
            enrichment,
            multisolute_pore_step,
        )
        from viva_fenics.fem_pore3d import build_pore3d_mesh

        cfg = self.config
        domain, ftags, markers = build_pore3d_mesh(
            length=cfg["length"], chamber_radius=cfg["chamber_radius"],
            throat_radius=cfg["throat_radius"], h_throat=cfg["h_throat"],
            h_chamber=cfg["h_chamber"],
        )
        V = fem.functionspace(domain, ("Lagrange", 1))
        n = V.dofmap.index_map.size_local
        z = V.tabulate_dof_coordinates()[:, 2]
        # hot inlet (z=0) -> cold outlet; Soret (S_T>0) concentrates cold-side.
        T = (330.0) - float(cfg["delta_T"]) * (z / float(cfg["length"]))

        specs = []
        for s in cfg["species"]:
            D = float(s["D"])
            # peclet sets the flow via w = Pe * D_ref / L (D_ref = this species'
            # D so each solute's own advective Peclet is the configured value);
            # the shared flow uses the mixture's mean D so one velocity drives
            # all solutes (they differ only in D and S_T).
            specs.append({
                "name": s["name"], "dissolved": np.zeros(n), "adsorbed": np.zeros(n),
                "D": D, "soret": float(s.get("soret", 0.0)),
                "inlet": s.get("inlet", 1.0),
                "k_ads": float(s.get("k_ads", 0.0)), "k_des": float(s.get("k_des", 0.0)),
                "k_deg": float(s.get("k_deg", 0.0)),
            })
        D_mean = float(np.mean([s["D"] for s in specs]))
        w = float(cfg["peclet"]) * D_mean / float(cfg["length"])

        for _ in range(int(cfg["n_steps"])):
            specs = multisolute_pore_step(
                domain, V, ftags, markers, specs, dt=float(cfg["dt"]),
                velocity=(0.0, 0.0, w), temperature_array=T,
            )

        names = [s["name"] for s in specs]
        enr = np.array([enrichment(domain, V, ftags, markers, s["dissolved"])
                        for s in specs], dtype=float)
        peak = np.array([float(np.max(s["dissolved"])) for s in specs], dtype=float)
        # pairwise spatial co-localization of the dissolved fields
        corrs = []
        for a, b in itertools.combinations(specs, 2):
            c = colocalization(a["dissolved"], b["dissolved"])
            if np.isfinite(c):
                corrs.append(c)
        coloc_min = float(np.min(corrs)) if corrs else float("nan")
        coloc_mean = float(np.mean(corrs)) if corrs else float("nan")

        self._species_names = names  # retained for callers that want the order
        return {
            "enrichment": enr,
            "peak": peak,
            "colocalization_min": coloc_min,
            "colocalization_mean": coloc_mean,
            "n_species": len(specs),
        }
