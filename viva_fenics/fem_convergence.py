"""Mesh- and time-step convergence for the 3-D pore transport solve, and the
**trustworthy-Peclet envelope** it defines.

The full modeling plan's rigor rule for the spatial face (§4.5): *no
acceptance gate outside demonstrated solver convergence*. A convection-
dominated pore solve develops an accumulation boundary layer whose thickness
scales like ``L / Pe``; once that layer is thinner than the local mesh
spacing the resolved field is numerical-diffusion-limited and the reported
enrichment is a mesh artifact, not physics. The *trustworthy-Peclet
envelope* is the range of Peclet number over which the readout is
demonstrably converged -- successive mesh refinements (and time-step
halvings) agree to within a tolerance -- so a study may only place a gate
inside it.

Everything here drives the REAL dolfinx solve on the real 3-D pore
(:mod:`viva_fenics.fem_pore3d` + :mod:`viva_fenics.fem_multisolute`); the
"convergence" is measured self-consistently (Richardson-style, readout vs
the next-finer level) and, where an analytic limit exists, against it.

Requires dolfinx; runs under viva-fenics's pixi env.
"""
from __future__ import annotations

import numpy as np
from dolfinx import fem

from .fem_multisolute import enrichment, multisolute_pore_step
from .fem_pore3d import build_pore3d_mesh


def run_pore_enrichment(
    *, peclet, soret=0.1, length=1.0, chamber_radius=0.4, throat_radius=0.12,
    h_throat=0.04, h_chamber=0.10, dt=0.02, n_steps=120, diffusivity=0.05,
):
    """Run one single-species Soret pore solve to (near) steady state and return
    its enrichment factor plus the mesh size.

    ``peclet`` sets the advective flow via ``w = peclet * D / length`` (so Pe is
    ``w L / D`` by construction); a hot inlet / cold outlet linear temperature
    field drives the Soret drift toward the cold (outlet) side. ``h_throat``
    controls the mesh fineness at the constriction (the convergence knob).

    Returns a dict: ``enrichment``, ``dofs``, ``h_throat``, ``dt``, ``peclet``,
    ``n_cells``.
    """
    domain, ftags, markers = build_pore3d_mesh(
        length=length, chamber_radius=chamber_radius, throat_radius=throat_radius,
        h_throat=h_throat, h_chamber=h_chamber,
    )
    V = fem.functionspace(domain, ("Lagrange", 1))
    n = V.dofmap.index_map.size_local
    coords = V.tabulate_dof_coordinates()
    z = coords[:, 2]
    # hot inlet (z=0) -> cold outlet (z=length); grad(T) along +z so Soret
    # (S_T > 0) concentrates toward the cold outlet/throat.
    T = 330.0 - 30.0 * (z / length)
    w = peclet * diffusivity / length
    species = [{
        "name": "tracer", "dissolved": np.zeros(n), "adsorbed": np.zeros(n),
        "D": diffusivity, "soret": soret, "inlet": 1.0,
    }]
    for _ in range(int(n_steps)):
        species = multisolute_pore_step(
            domain, V, ftags, markers, species, dt=dt,
            velocity=(0.0, 0.0, float(w)), temperature_array=T,
        )
    e = enrichment(domain, V, ftags, markers, species[0]["dissolved"])
    return {
        "enrichment": float(e), "dofs": int(n),
        "n_cells": int(domain.topology.index_map(domain.topology.dim).size_local),
        "h_throat": float(h_throat), "dt": float(dt), "peclet": float(peclet),
    }


def self_convergence(values, rtol=0.05):
    """Given a readout measured at successively finer resolutions (coarse→fine),
    return the successive relative changes and whether the sequence has
    converged (the last relative change is below ``rtol``). ``rel_changes[i] =
    |v[i+1]-v[i]| / max(|v[i+1]|, eps)``. Converged means the finest step moved
    the readout by less than ``rtol`` -- further refinement would not change the
    answer, so the readout is trustworthy at this resolution."""
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return {"rel_changes": [], "converged": False, "final_change": np.nan}
    rel = np.abs(np.diff(v)) / np.maximum(np.abs(v[1:]), 1e-30)
    final = float(rel[-1])
    return {
        "rel_changes": [float(r) for r in rel],
        "final_change": final,
        "converged": bool(final < rtol and np.isfinite(final)),
    }


def mesh_convergence(*, peclet, h_throats=(0.08, 0.05, 0.03), rtol=0.05, **kw):
    """Refine the throat mesh (coarse→fine) at a fixed Peclet and report whether
    the enrichment readout converges. Returns the per-level records, the
    ``self_convergence`` summary of the enrichment sequence, and ``converged``."""
    records = [run_pore_enrichment(peclet=peclet, h_throat=h, **kw)
               for h in h_throats]
    conv = self_convergence([r["enrichment"] for r in records], rtol=rtol)
    return {"peclet": float(peclet), "levels": records,
            "enrichment_convergence": conv, "converged": conv["converged"]}


def dt_convergence(*, peclet, dts=(0.04, 0.02, 0.01), n_steps_at_coarsest=60,
                   rtol=0.05, h_throat=0.04, **kw):
    """Halve the time step (coarse→fine) at a fixed Peclet + mesh and report
    whether the enrichment readout converges. ``n_steps`` is scaled up as ``dt``
    shrinks so every level integrates to the same physical time."""
    t_end = dts[0] * n_steps_at_coarsest
    records = []
    for dt in dts:
        n_steps = int(round(t_end / dt))
        records.append(run_pore_enrichment(peclet=peclet, dt=dt, n_steps=n_steps,
                                           h_throat=h_throat, **kw))
    conv = self_convergence([r["enrichment"] for r in records], rtol=rtol)
    return {"peclet": float(peclet), "levels": records,
            "enrichment_convergence": conv, "converged": conv["converged"]}


def trustworthy_peclet_envelope(
    *, peclets=(2.0, 5.0, 10.0, 20.0, 40.0),
    h_throats=(0.08, 0.05, 0.03), rtol=0.05, **kw,
):
    """Sweep Peclet and, at each, run a mesh-refinement convergence study; the
    envelope is the contiguous low-Peclet band where the enrichment readout is
    mesh-converged.

    Returns ``{per_peclet: [...], converged_peclets: [...],
    peclet_max_trustworthy: float}`` where ``peclet_max_trustworthy`` is the
    largest Peclet for which this and every lower swept Peclet converged (so a
    gate at or below it is inside demonstrated convergence). ``None`` if even the
    lowest Peclet did not converge at the given resolutions."""
    per = [mesh_convergence(peclet=pe, h_throats=h_throats, rtol=rtol, **kw)
           for pe in peclets]
    ordered = sorted(per, key=lambda r: r["peclet"])
    envelope = None
    converged_pes = []
    for r in ordered:
        if r["converged"]:
            converged_pes.append(r["peclet"])
            envelope = r["peclet"]
        else:
            break  # first Peclet that fails to converge caps the envelope
    return {
        "per_peclet": per,
        "converged_peclets": converged_pes,
        "peclet_max_trustworthy": envelope,
        "rtol": rtol,
        "h_throats": list(h_throats),
    }
