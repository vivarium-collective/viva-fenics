"""Composite generator for the multisolute 3-D pore reference: a single
``MultisolutePoreStep`` wired to a RAM emitter, so the dashboard's Composites
tab / a bare workspace run can build and execute the full multisolute pore
simulation at one parameterization.

The real study -- the co-localization phase map over Peclet x mixture, with
the trustworthy-Peclet envelope and the reduced surrogate for large ensembles
-- lives in the viva-genesis ``localization`` face (Phase 4), which drives
``viva_fenics.fem_convergence`` / ``fem_multisolute`` directly for the richer
sweep. This generator exposes a real, buildable composite id for the solver.
"""
from __future__ import annotations

from viva_superpowers.composite_generator import composite_generator


@composite_generator(
    name="multisolute_pore",
    description=(
        "High-fidelity 3-D reference for multisolute thermophoretic "
        "localization: a representative converging-diverging pore (dolfinx "
        "FEM, gmsh tetrahedral mesh) in which several solutes are transported "
        "SIMULTANEOUSLY, each with its own diffusivity, Soret coefficient, "
        "adsorption and degradation, under a shared flow + hot-inlet/cold-"
        "outlet temperature field. Reports per-species trap enrichment "
        "(peak/feed) and the mixture's worst-pair spatial co-localization -- "
        "whether the chemically-necessary mixture concentrates together or "
        "fractionates. Runs under viva-fenics's pixi env (dolfinx)."
    ),
    parameters={
        "peclet": {"type": "number", "default": 8.0,
                   "description": "Advective Peclet number (w L / D_mean) of the pore flow"},
        "throat_radius": {"type": "number", "default": 0.12,
                          "description": "Pore throat (constriction) radius; < chamber_radius"},
        "delta_T": {"type": "number", "default": 30.0,
                    "description": "Hot-inlet to cold-outlet temperature drop driving Soret"},
        "n_steps": {"type": "integer", "default": 120,
                    "description": "Backward-Euler steps to near steady state"},
    },
)
def multisolute_pore(core=None, *, peclet=8.0, throat_radius=0.12, delta_T=30.0, n_steps=120):
    emit = {
        "enrichment": "array[float]",
        "peak": "array[float]",
        "colocalization_min": "float",
        "colocalization_mean": "float",
        "n_species": "integer",
    }
    ports = {
        "enrichment": ["stores", "enrichment"],
        "peak": ["stores", "peak"],
        "colocalization_min": ["stores", "colocalization_min"],
        "colocalization_mean": ["stores", "colocalization_mean"],
        "n_species": ["stores", "n_species"],
    }
    return {
        "pore": {
            "_type": "step",
            "address": "local:MultisolutePoreStep",
            "config": {
                "peclet": peclet,
                "throat_radius": throat_radius,
                "delta_T": delta_T,
                "n_steps": n_steps,
            },
            "inputs": {},
            "outputs": ports,
        },
        "stores": {
            "enrichment": [],
            "peak": [],
            "colocalization_min": 0.0,
            "colocalization_mean": 0.0,
            "n_species": 0,
        },
        "emitter": {
            "_type": "step",
            "address": "local:RAMEmitter",
            "config": {"emit": emit},
            "inputs": ports,
        },
    }
