"""Multisolute reactive transport on a tagged pore mesh (dolfinx).

The full modeling plan's Stage-III decisive test (§4.5) is not whether *one*
tracer concentrates, but whether the chemically-necessary **mixture** of
prebiotic solutes co-localizes or fractionates: phosphate, ammonium,
formate, acetate, pyruvate, amino-acid and nucleotide precursors, and
amphiphiles -- each with its own diffusivity ``D``, Soret (thermophoretic)
coefficient ``S_T``, surface adsorption, and degradation. Spatial
fractionation may be a positive concentrating mechanism or a fatal
incompatibility; this module is the solver that lets a study report which.

The physics, per species, on the SAME pore geometry / flow field /
temperature field:

* **Transport** -- one backward-Euler advection-diffusion-Soret solve
  (:func:`pore_transport_step`), identical weak form to
  :func:`viva_fenics.fem.advection_diffusion_step` but with the inlet
  Dirichlet concentration applied on the ``inlet`` facet tag (the pore's
  ``z=0`` cap) rather than the unit square's ``x=0`` face.
* **Adsorption** -- a linear reversible exchange dissolved <-> adsorbed with
  rates ``k_ads``/``k_des``, integrated analytically over the step so the
  dissolved+adsorbed pool is conserved exactly (adsorption moves mass, it
  does not destroy it).
* **Degradation** -- first-order decay of the dissolved pool,
  ``d <- d * exp(-k_deg * dt)``.

Species are coupled only through the shared flow and temperature fields (no
cross-diffusion), so the mixture is advanced as N independent single-field
solves -- the physically-correct, matrix-free decomposition. Co-localization
is then an *emergent* comparison of where each species ends up, not an
assumption baked into a coupled matrix.

Requires dolfinx (conda-only); runs under viva-fenics's pixi env.
"""
from __future__ import annotations

import itertools

import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI  # noqa: F401  (imported for parity / COMM in callers)

_prefix_counter = itertools.count()


def pore_transport_step(
    domain, V, facet_tags, markers, u_n_array, source_array, dt, D,
    velocity=None, soret=None, temperature_array=None,
    inlet_value=None, supg=True,
):
    """One backward-Euler advection-diffusion-Soret step on a tagged pore mesh.

    Same weak form as :func:`viva_fenics.fem.advection_diffusion_step`
    (diffusion + optional divergence-free advection with transient-consistent
    SUPG + optional conservative Soret drift), but the optional Dirichlet BC is
    the **inlet concentration** applied on ``facet_tags.find(markers["inlet"])``
    -- the pore's ``z=0`` cap -- so it works on an arbitrary gmsh pore geometry
    (the base solver pins the unit square's ``x=0`` face, which is meaningless
    on a 3-D pore). ``inlet_value=None`` leaves the inlet a natural (zero-flux)
    boundary.

    Returns the nodal solution array after the step.
    """
    gdim = domain.geometry.dim
    u_n = fem.Function(V)
    u_n.x.array[:] = u_n_array
    source = fem.Function(V)
    source.x.array[:] = source_array

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(domain, float(dt))
    D_c = fem.Constant(domain, float(D))

    w_eff = None
    if velocity is not None:
        degree = V.element.basix_element.degree
        W = fem.functionspace(domain, ("Lagrange", degree, (gdim,)))
        w_fn = fem.Function(W)
        if isinstance(velocity, (tuple, list)):
            comps = [float(c) for c in velocity]
            if len(comps) != gdim:
                raise ValueError(
                    f"constant velocity needs {gdim} components, got {len(comps)}"
                )
            w_fn.interpolate(lambda x: np.stack([np.full(x.shape[1], c) for c in comps]))
        else:
            w_fn.x.array[:] = np.asarray(velocity, dtype=float)
        w_eff = w_fn

    a = (u * v + dt_c * D_c * ufl.dot(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n + dt_c * source) * v * ufl.dx

    if soret is not None and temperature_array is not None:
        T_fn = fem.Function(V)
        T_fn.x.array[:] = np.asarray(temperature_array, dtype=float)
        soret_c = fem.Constant(domain, float(soret))
        a += dt_c * D_c * soret_c * u * ufl.dot(ufl.grad(T_fn), ufl.grad(v)) * ufl.dx

    if w_eff is not None:
        a += dt_c * ufl.dot(w_eff, ufl.grad(u)) * v * ufl.dx
        if supg:
            h = ufl.CellDiameter(domain)
            wnorm = ufl.sqrt(ufl.dot(w_eff, w_eff) + 1e-30)
            tau = h / (2.0 * wnorm)
            r_impl = u + dt_c * ufl.dot(w_eff, ufl.grad(u))
            r_expl = u_n + dt_c * source
            a += tau * ufl.dot(w_eff, ufl.grad(v)) * r_impl * ufl.dx
            L += tau * ufl.dot(w_eff, ufl.grad(v)) * r_expl * ufl.dx

    bcs = []
    if inlet_value is not None:
        tdim = domain.topology.dim
        inlet_facets = facet_tags.find(markers["inlet"])
        dofs = fem.locate_dofs_topological(V, tdim - 1, inlet_facets)
        bc_fn = fem.Function(V)
        bc_fn.x.array[:] = float(inlet_value)
        bcs.append(fem.dirichletbc(bc_fn, dofs))

    prefix = f"viva_fenics_pore_{next(_prefix_counter)}_"
    problem = LinearProblem(
        a, L, bcs=bcs, petsc_options_prefix=prefix,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    uh = problem.solve()
    return uh.x.array.copy()


def adsorb_degrade(dissolved, adsorbed, dt, k_ads=0.0, k_des=0.0, k_deg=0.0):
    """Apply one step of reversible adsorption + first-order degradation to the
    (dissolved, adsorbed) pools, node-wise and analytically.

    Adsorption is a linear two-pool exchange ``d' = -k_ads d + k_des a``,
    ``a' = +k_ads d - k_des a`` whose total ``d + a`` is invariant; it relaxes
    toward the equilibrium split ``a_eq = total * k_ads/(k_ads+k_des)`` at rate
    ``k_ads + k_des``, so the exact update over ``dt`` is
    ``a(t+dt) = a_eq + (a - a_eq) exp(-(k_ads+k_des) dt)``. Degradation then
    removes from the dissolved pool only (adsorbed species are protected on the
    mineral surface -- the usual prebiotic-chemistry assumption):
    ``d <- d exp(-k_deg dt)``. Returns ``(dissolved, adsorbed)``.
    """
    d = np.asarray(dissolved, dtype=float).copy()
    a = np.asarray(adsorbed, dtype=float).copy()
    ktot = k_ads + k_des
    if ktot > 0.0:
        total = d + a
        a_eq = total * (k_ads / ktot)
        a = a_eq + (a - a_eq) * np.exp(-ktot * dt)
        d = total - a
    if k_deg > 0.0:
        d = d * np.exp(-k_deg * dt)
    return d, a


def multisolute_pore_step(domain, V, facet_tags, markers, species, dt,
                          velocity=None, temperature_array=None, supg=True):
    """Advance a list of solute species one step on the shared pore.

    ``species`` is a list of dicts, each with:
        name      : str
        dissolved : nodal array (updated in the returned copy)
        adsorbed  : nodal array
        D         : diffusivity
        soret     : Soret coefficient S_T (1/K); 0 -> no thermophoresis
        inlet     : inlet Dirichlet concentration (None -> zero-flux inlet)
        k_ads, k_des, k_deg : adsorption / desorption / degradation rates (optional)

    Each species is transported with its OWN D and S_T through the shared
    ``velocity`` and ``temperature_array``, then its adsorption + degradation is
    applied. Returns a new list of species dicts with updated dissolved/adsorbed
    arrays (inputs are not mutated).
    """
    out = []
    for s in species:
        d_star = pore_transport_step(
            domain, V, facet_tags, markers,
            s["dissolved"], np.zeros_like(s["dissolved"]), dt, s["D"],
            velocity=velocity, soret=s.get("soret", 0.0) or None,
            temperature_array=temperature_array,
            inlet_value=s.get("inlet"), supg=supg,
        )
        d, a = adsorb_degrade(
            d_star, s["adsorbed"], dt,
            k_ads=s.get("k_ads", 0.0), k_des=s.get("k_des", 0.0),
            k_deg=s.get("k_deg", 0.0),
        )
        rec = dict(s)
        rec["dissolved"] = d
        rec["adsorbed"] = a
        out.append(rec)
    return out


def enrichment(domain, V, facet_tags, markers, field):
    """Thermophoretic-trap enrichment factor of a nodal ``field``: the peak
    concentration anywhere in the pore over the inlet feed concentration. This
    is the standard trap readout (the og-03 ``exp(Pe)`` column measures the same
    thing: how far above feed the accumulation region climbs). ``> 1`` means the
    pore concentrates the species; ``1`` means it is merely advected through at
    feed level. When there is no through-inlet feed (pure Soret redistribution)
    it falls back to peak/mean so an unfed run still yields a finite ratio."""
    tdim = domain.topology.dim
    field = np.asarray(field, dtype=float)
    if field.size == 0:
        return np.nan

    inlet_facets = facet_tags.find(markers["inlet"])
    inlet_dofs = fem.locate_dofs_topological(V, tdim - 1, inlet_facets)
    feed = float(np.mean(field[inlet_dofs])) if len(inlet_dofs) else np.nan
    peak = float(np.max(field))
    if feed and abs(feed) > 1e-30:
        return peak / feed
    base = float(np.mean(field))
    return peak / base if base and abs(base) > 1e-30 else np.nan


def colocalization(field_a, field_b):
    """How much two species' spatial distributions overlap -- the Pearson
    correlation of their nodal concentration fields. ``+1`` = they concentrate
    in exactly the same regions (co-localize), ``0`` = independent, ``< 0`` =
    they anti-localize (where one concentrates the other is depleted -- spatial
    fractionation, the mixture incompatibility the plan asks about). Returns NaN
    if either field is spatially uniform (no distribution to correlate)."""
    a = np.asarray(field_a, dtype=float).ravel()
    b = np.asarray(field_b, dtype=float).ravel()
    if a.size != b.size or a.size < 2:
        return np.nan
    if np.std(a) < 1e-30 or np.std(b) < 1e-30:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])
