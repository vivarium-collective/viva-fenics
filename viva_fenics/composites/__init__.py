"""viva-fenics composite generators (imported for @composite_generator side effects)."""

from . import poisson  # noqa: F401
from . import diffusion  # noqa: F401
from . import reaction_diffusion as _reaction_diffusion_module  # noqa: F401
from . import convergence  # noqa: F401
from . import flow as _flow_module  # noqa: F401
from .poisson import poisson_baseline
from .diffusion import transient_diffusion
from .reaction_diffusion import reaction_diffusion
from .convergence import mesh_convergence
from .flow import navier_stokes

__all__ = [
    "poisson_baseline",
    "transient_diffusion",
    "reaction_diffusion",
    "mesh_convergence",
    "navier_stokes",
]
