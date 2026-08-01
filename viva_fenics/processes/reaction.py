"""Pure-numpy process-bigraph reaction Processes, coupled to
``DiffusionProcess`` purely through shared bigraph stores -- neither this
module nor ``diffusion.py`` has any knowledge of the other.

``LogisticReactionProcess`` implements the logistic growth term
``r*u*(1 - u/K)`` of the Fisher-KPP equation (a single species; see
``viva_fenics.composites.reaction_diffusion`` for its 2-process coupling).
``GrayScottReactionProcess`` implements the two-species Gray-Scott feed/kill
kinetics that, coupled to TWO ``DiffusionProcess`` instances (one per
species), produce a genuine Turing instability (see
``viva_fenics.composites.turing_patterns`` for its 3-process coupling).
``LinearDegradationProcess`` implements the first-order decay term ``-k*c``
of the Source-Diffusion-Degradation (SDD) morphogen-gradient model, coupled
to a single ``DiffusionProcess`` (with a Dirichlet source boundary) to
produce a genuine exponential gradient (see
``viva_fenics.composites.morphogen_gradient``).
"""

from __future__ import annotations

import numpy as np
from process_bigraph import Process


class LogisticReactionProcess(Process):
    """Logistic reaction term for Fisher-KPP: ``r*u*(1 - u/K)``.

    Inputs
    ------
    solution : array[float]
        Current absolute nodal field (the same store ``DiffusionProcess``
        reads/writes as its own "solution").

    Outputs
    -------
    source : overwrite[array[float]]
        The instantaneous reaction RATE field for this tick. This is an
        ``overwrite`` (replace), not a bare additive ``array[float]``: it is
        not a delta to accumulate, it is a fresh per-tick rate reading that
        must *replace* the shared ``stores.source`` store each tick.
        ``DiffusionProcess`` treats "source" as a per-unit-time rate and
        multiplies it by dt internally (see ``fem.diffusion_step``'s
        ``(u_n + dt*source)*v*dx``), so this process must NOT also multiply
        by ``interval`` -- doing so would double-apply the timestep. If
        "source" were instead additive, repeated ticks would accumulate
        ``sum_i rate_i`` in the shared store instead of reporting just the
        current tick's rate, and the coupled field would blow up well past
        K (the same class of bug the Task 3 `integral` sensor had before it
        was made ``overwrite[float]``).
    """

    config_schema = {
        "r": {"_type": "float", "_default": 1.0},
        "K": {"_type": "float", "_default": 1.0},
    }

    def inputs(self):
        return {"solution": "array[float]"}

    def outputs(self):
        return {"source": "overwrite[array[float]]"}

    def update(self, state, interval):
        u = np.asarray(state["solution"], dtype=float)
        r = self.config["r"]
        K = self.config["K"]
        source = r * u * (1 - u / K)
        return {"source": source}


class GrayScottReactionProcess(Process):
    """Gray-Scott reaction term coupling TWO diffusing species U, V:

        source_u = -U*V^2 + F*(1-U)
        source_v = +U*V^2 - (F+k)*V

    Like ``LogisticReactionProcess``, this process has NO knowledge of FEM or
    of ``DiffusionProcess`` -- it only reads "u"/"v" fields and writes
    "source_u"/"source_v" rate fields. The Turing-pattern coupling (see
    ``viva_fenics.composites.turing_patterns``) is entirely the document
    wiring: TWO independent ``DiffusionProcess`` instances (one per species,
    with differential diffusivities ``Du`` > ``Dv``) each read/write their own
    species' "solution"/"source" ports against ``stores.u``/``stores.source_u``
    and ``stores.v``/``stores.source_v`` respectively, while this process
    reads BOTH ``stores.u``/``stores.v`` and writes BOTH
    ``stores.source_u``/``stores.source_v`` -- three independently-authored
    processes, coupled purely by sharing those four stores. That three-way
    coupling (not two, as in ``LogisticReactionProcess``'s single-species
    Fisher-KPP pairing) is what makes a genuine 2D Turing instability -- and
    the spot/stripe/labyrinth patterns it produces -- emerge from
    composition, not from any one process implementing Gray-Scott itself.

    Inputs
    ------
    u, v : array[float]
        Current absolute nodal fields for each species (the same stores the
        two ``DiffusionProcess`` instances read/write as their own
        "solution").

    Outputs
    -------
    source_u, source_v : overwrite[array[float]]
        The instantaneous per-species reaction RATE fields for this tick.
        ``overwrite`` (replace), not additive -- same rationale as
        ``LogisticReactionProcess.outputs``'s "source": each
        ``DiffusionProcess`` multiplies its own source rate by dt internally
        (``fem.diffusion_step``'s ``(u_n + dt*source)*v*dx``), so this
        process must NOT also scale by ``interval`` or the reaction would be
        double-applied; and each store must be REPLACED every tick, not
        accumulated, or the coupled fields would blow up instead of
        self-organizing into a bounded Turing pattern.
    """

    config_schema = {
        "F": {"_type": "float", "_default": 0.037},
        "k": {"_type": "float", "_default": 0.06},
    }

    def inputs(self):
        return {"u": "array[float]", "v": "array[float]"}

    def outputs(self):
        return {
            "source_u": "overwrite[array[float]]",
            "source_v": "overwrite[array[float]]",
        }

    def update(self, state, interval):
        u = np.asarray(state["u"], dtype=float)
        v = np.asarray(state["v"], dtype=float)
        F = self.config["F"]
        k = self.config["k"]

        uvv = u * v * v
        source_u = -uvv + F * (1.0 - u)
        source_v = uvv - (F + k) * v
        return {"source_u": source_u, "source_v": source_v}


class LinearDegradationProcess(Process):
    """First-order degradation term for the Source-Diffusion-Degradation
    (SDD) morphogen-gradient model: ``source = -k*c``.

    Coupled to ``DiffusionProcess`` (with a Dirichlet source boundary c=c0
    at x=0, see ``DiffusionProcess``'s ``apply_boundary``/``boundary_value``
    config) purely through shared bigraph stores -- this process has NO
    knowledge of FEM, of DiffusionProcess, or of the boundary condition. The
    steady state of the composed system (``0 = D*laplacian(c) - k*c``, with
    c(0)=c0) is the classic exponential morphogen gradient
    ``c(x) = c0*exp(-x/lambda)``, decay length ``lambda = sqrt(D/k)`` -- see
    ``viva_fenics.composites.morphogen_gradient`` for the composition.

    Inputs
    ------
    solution : array[float]
        Current absolute nodal field (the same store ``DiffusionProcess``
        reads/writes as its own "solution").

    Outputs
    -------
    source : overwrite[array[float]]
        The instantaneous degradation RATE field for this tick (always
        <= 0 for c >= 0). ``overwrite`` (replace), not additive -- same
        rationale as ``LogisticReactionProcess.outputs``'s "source":
        ``DiffusionProcess`` multiplies "source" by dt internally, so this
        process must not also scale by ``interval``, and the store must be
        REPLACED each tick (a fresh rate reading), not accumulated.
    """

    config_schema = {
        "k": {"_type": "float", "_default": 1.0},
    }

    def inputs(self):
        return {"solution": "array[float]"}

    def outputs(self):
        return {"source": "overwrite[array[float]]"}

    def update(self, state, interval):
        c = np.asarray(state["solution"], dtype=float)
        k = self.config["k"]
        return {"source": -k * c}
