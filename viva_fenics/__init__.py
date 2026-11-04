"""Process-bigraph wrapper for FEniCSx (dolfinx).

Most of this package needs dolfinx (conda-only, PETSc/MPI). The
``spatial_transport`` face adapters, however, keep their dolfinx import lazy
(inside ``update()``), so they can be imported and *registered* in a consumer
workspace's dolfinx-free ``uv`` venv -- only *running* one there raises a clear
"run under pixi" error. To keep ``import viva_fenics`` working in that venv, the
spatial-transport classes are imported unconditionally below and the
dolfinx-dependent imports are guarded: without dolfinx, ``import viva_fenics``
still succeeds and exposes those two classes, while the heavy FEM processes /
composites / ``build_core`` are simply absent. Only a missing FEniCSx-stack
module is tolerated; any other ImportError propagates.
"""

import importlib.util

# dolfinx-free: importable anywhere (dolfinx import stays lazy inside update()).
from viva_fenics.processes.spatial_transport import (
    FenicsReactiveTransportProcess,
    FenicsThermophoreticTrapProcess,
)

__all__ = [
    "FenicsReactiveTransportProcess",
    "FenicsThermophoreticTrapProcess",
]

# The conda FEniCSx stack (pixi.toml): the only modules allowed to be absent.
_FENICSX_MODULES = frozenset({"dolfinx", "ufl", "basix", "ffcx", "mpi4py", "petsc4py", "gmsh"})

try:  # the dolfinx-backed FEM stack (present under pixi, absent in a uv .venv)
    from viva_fenics import viz
    from viva_fenics.processes.poisson import PoissonSolverStep
    from viva_fenics.processes.diffusion import DiffusionProcess
    from viva_fenics.processes.reaction import LogisticReactionProcess, LinearDegradationProcess
    from viva_fenics.processes.flow import NavierStokesProcess, CylinderFlowProcess
    from viva_fenics.processes.moving_boundary import MovingBoundaryProcess
    from viva_fenics.processes.peristalsis import PeristalticWallProcess, PeristalticFlowProcess
    from viva_fenics.processes.amr import AdaptiveRefinementStep
    from viva_fenics.composites.poisson import poisson_baseline, high_order_verification
    from viva_fenics.composites.diffusion import transient_diffusion
    from viva_fenics.composites.reaction_diffusion import reaction_diffusion
    from viva_fenics.composites.morphogen_gradient import morphogen_gradient
    from viva_fenics.composites.convergence import mesh_convergence
    from viva_fenics.composites.flow import navier_stokes, vortex_street
    from viva_fenics.composites.moving_boundary import moving_boundary
    from viva_fenics.composites.peristalsis import peristalsis
    from viva_fenics.composites.amr import adaptive_refinement
    from viva_fenics.core import build_core, register_processes
except ImportError as exc:  # pragma: no cover - dolfinx not installed (consumer uv venv)
    # Only an absent conda FEniCSx stack is expected here. Any other missing
    # module is an undeclared dependency, and a FEniCSx module that is installed
    # but fails to import (e.g. an API outside the pinned dolfinx range) is a
    # broken environment: raise both rather than silently drop every class and
    # composite below.
    root = (exc.name or "").split(".")[0]
    if root not in _FENICSX_MODULES or importlib.util.find_spec(root) is not None:
        raise
    _DOLFINX_AVAILABLE = False
else:
    _DOLFINX_AVAILABLE = True
    __all__ += [
        "PoissonSolverStep",
        "DiffusionProcess",
        "LogisticReactionProcess",
        "LinearDegradationProcess",
        "NavierStokesProcess",
        "CylinderFlowProcess",
        "MovingBoundaryProcess",
        "PeristalticWallProcess",
        "PeristalticFlowProcess",
        "AdaptiveRefinementStep",
        "poisson_baseline",
        "high_order_verification",
        "transient_diffusion",
        "reaction_diffusion",
        "morphogen_gradient",
        "mesh_convergence",
        "navier_stokes",
        "vortex_street",
        "moving_boundary",
        "peristalsis",
        "adaptive_refinement",
        "build_core",
        "register_processes",
        "viz",
    ]
