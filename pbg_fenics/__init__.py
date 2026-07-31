"""Process-bigraph wrapper for FEniCSx (dolfinx)."""

from pbg_fenics import viz
from pbg_fenics.processes.poisson import PoissonSolverStep
from pbg_fenics.processes.diffusion import DiffusionProcess
from pbg_fenics.processes.reaction import LogisticReactionProcess
from pbg_fenics.composites.poisson import poisson_baseline
from pbg_fenics.composites.diffusion import transient_diffusion
from pbg_fenics.composites.reaction_diffusion import reaction_diffusion
from pbg_fenics.composites.convergence import mesh_convergence
from pbg_fenics.core import build_core, register_processes

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

