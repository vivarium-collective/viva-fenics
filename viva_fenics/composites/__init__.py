"""viva-fenics composite generators (imported for @composite_generator side effects)."""

from . import poisson  # noqa: F401
from . import diffusion  # noqa: F401
from . import reaction_diffusion as _reaction_diffusion_module  # noqa: F401
from . import turing_patterns as _turing_patterns_module  # noqa: F401
from . import morphogen_gradient as _morphogen_gradient_module  # noqa: F401
from . import convergence  # noqa: F401
from . import flow as _flow_module  # noqa: F401
from . import moving_boundary as _moving_boundary_module  # noqa: F401
from . import complex_geometry as _complex_geometry_module  # noqa: F401
from . import amr as _amr_module  # noqa: F401
from .poisson import poisson_baseline, high_order_verification
from .diffusion import transient_diffusion
from .reaction_diffusion import reaction_diffusion
from .turing_patterns import turing_patterns
from .morphogen_gradient import morphogen_gradient
from .convergence import mesh_convergence
from .flow import navier_stokes, vortex_street
from .moving_boundary import moving_boundary
from .complex_geometry import complex_geometry
from .amr import adaptive_refinement

__all__ = [
    "poisson_baseline",
    "high_order_verification",
    "transient_diffusion",
    "reaction_diffusion",
    "turing_patterns",
    "morphogen_gradient",
    "mesh_convergence",
    "navier_stokes",
    "vortex_street",
    "moving_boundary",
    "complex_geometry",
    "adaptive_refinement",
]
