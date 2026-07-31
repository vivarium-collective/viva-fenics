"""LogisticReactionProcess: a pure-numpy process-bigraph Process implementing
the logistic growth term ``r*u*(1 - u/K)`` of the Fisher-KPP equation.

This process has NO knowledge of FEM or of DiffusionProcess -- it only reads
a "solution" field and writes a "source" field. Coupling to
``DiffusionProcess`` (which reads "source" as its own input) happens purely
through the bigraph document wiring both processes to the same
``stores.source`` / ``stores.solution`` paths (see
``viva_fenics.composites.reaction_diffusion``), not by either process calling
into the other. That is the composability property this process exists to
demonstrate.
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
