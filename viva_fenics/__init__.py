"""Process-bigraph wrapper for FEniCSx (dolfinx)."""

from viva_fenics import viz
from viva_fenics.processes.poisson import PoissonSolverStep
from viva_fenics.processes.diffusion import DiffusionProcess
from viva_fenics.processes.reaction import LogisticReactionProcess, LinearDegradationProcess
from viva_fenics.processes.flow import NavierStokesProcess, CylinderFlowProcess
from viva_fenics.processes.moving_boundary import MovingBoundaryProcess
from viva_fenics.processes.peristalsis import PeristalticWallProcess, PeristalticFlowProcess
from viva_fenics.processes.complex_geometry import ComplexGeometryStep
from viva_fenics.processes.amr import AdaptiveRefinementStep
from viva_fenics.composites.poisson import poisson_baseline, high_order_verification
from viva_fenics.composites.diffusion import transient_diffusion
from viva_fenics.composites.reaction_diffusion import reaction_diffusion
from viva_fenics.composites.morphogen_gradient import morphogen_gradient
from viva_fenics.composites.convergence import mesh_convergence
from viva_fenics.composites.flow import navier_stokes, vortex_street
from viva_fenics.composites.moving_boundary import moving_boundary
from viva_fenics.composites.peristalsis import peristalsis
from viva_fenics.composites.complex_geometry import complex_geometry
from viva_fenics.composites.amr import adaptive_refinement
from viva_fenics.core import build_core, register_processes

__all__ = [
    "PoissonSolverStep",
    "DiffusionProcess",
    "LogisticReactionProcess",
    "LinearDegradationProcess",
    "NavierStokesProcess",
    "CylinderFlowProcess",
    "MovingBoundaryProcess",
    "PeristalticWallProcess",
    "PeristalticFlowProcess",
    "ComplexGeometryStep",
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
    "complex_geometry",
    "adaptive_refinement",
    "build_core",
    "register_processes",
    "viz",
]

