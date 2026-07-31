"""Workspace core builder.

The dashboard (and this repo's own composite generators) run each composite
in a subprocess/venv that calls this ``build_core()`` to obtain a core with
pbg-fenics's own types and process/step classes registered. Editable
installs are invisible to ``process_bigraph.allocate_core()``'s
distribution-keyed auto-discovery, so ``local:PoissonSolverStep`` /
``local:DiffusionProcess`` otherwise fail to resolve with "no link found at
address" unless a caller manually calls ``core.register_link(...)`` (as
tests/test_diffusion.py previously had to). Register them explicitly here
instead.
"""

from __future__ import annotations

from process_bigraph import allocate_core

from .processes.poisson import PoissonSolverStep
from .processes.diffusion import DiffusionProcess
from .processes.reaction import LogisticReactionProcess
from .types import register_types

# New processes (NavierStokesProcess, ...) just need an import above and an
# entry here.
_PROCESSES = (
    ("PoissonSolverStep", PoissonSolverStep),
    ("DiffusionProcess", DiffusionProcess),
    ("LogisticReactionProcess", LogisticReactionProcess),
)


def register_processes(core):
    """Register pbg-fenics's own Process/Step classes into ``core``."""
    for name, cls in _PROCESSES:
        core.register_link(name, cls)
    return core


def build_core(core=None):
    """Return a core with pbg-fenics's types and processes registered.

    Allocates a fresh core via ``allocate_core()`` when ``core`` is not
    given.
    """
    if core is None:
        core = allocate_core()
    register_types(core)
    register_processes(core)
    return core
