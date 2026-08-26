"""Skip the FEniCSx-backend tests when the dolfinx solver is unavailable.

viva-fenics wraps dolfinx (FEniCSx); a vanilla GitHub runner cannot build it, so
every test that imports the solver — directly, or via ``viva_fenics.core.build_core``
which imports the solver process modules — cannot run there. When dolfinx is
absent we SKIP those modules (they run in the repo's Docker image, where dolfinx
is present) and emit a clear notice, rather than failing CI. This keeps
workspace-ci honest: it runs the backend-free checks it can and states plainly
what it skipped — it does not pretend to have run a solve.
"""
from __future__ import annotations

try:
    import dolfinx  # noqa: F401
    _HAVE_BACKEND = True
except Exception:  # noqa: BLE001 — any import error means no usable backend
    _HAVE_BACKEND = False

# Modules that need the dolfinx backend (import it, or import viva_fenics.*,
# which pulls the solver process modules via build_core).
_BACKEND_TEST_MODULES = [
    "test_amr.py", "test_complex_geometry.py", "test_convergence.py",
    "test_core.py", "test_core_registration.py", "test_diffusion.py",
    "test_fem.py", "test_flow.py", "test_morphogen_gradient.py",
    "test_moving_boundary.py", "test_peristalsis.py", "test_poisson.py",
    "test_reaction_diffusion.py", "test_turing_patterns.py", "test_viz.py",
]

collect_ignore = [] if _HAVE_BACKEND else list(_BACKEND_TEST_MODULES)


def pytest_configure(config):
    if not _HAVE_BACKEND:
        config.issue_config_time_warning(
            __import__("pytest").PytestConfigWarning(
                "dolfinx (FEniCSx backend) unavailable — skipping "
                f"{len(_BACKEND_TEST_MODULES)} backend test module(s); run them "
                "in the repo's Docker image. Backend-free checks still run."
            ),
            stacklevel=1,
        )
