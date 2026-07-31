"""Process-bigraph wrapper for FEniCSx (dolfinx)."""

from viva_fenics import viz
from viva_fenics.processes.poisson import PoissonSolverStep
from viva_fenics.processes.diffusion import DiffusionProcess
from viva_fenics.processes.reaction import LogisticReactionProcess
from viva_fenics.composites.poisson import poisson_baseline
from viva_fenics.composites.diffusion import transient_diffusion
from viva_fenics.composites.reaction_diffusion import reaction_diffusion
from viva_fenics.composites.convergence import mesh_convergence
from viva_fenics.core import build_core, register_processes

__all__ = [
    "PoissonSolverStep",
    "DiffusionProcess",
    "LogisticReactionProcess",
    "poisson_baseline",
    "transient_diffusion",
    "reaction_diffusion",
    "mesh_convergence",
    "build_core",
    "register_processes",
    "viz",
]

