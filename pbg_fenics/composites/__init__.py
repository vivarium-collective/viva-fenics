"""pbg-fenics composite generators (imported for @composite_generator side effects)."""

from . import poisson  # noqa: F401
from .poisson import poisson_baseline

__all__ = ["poisson_baseline"]
