"""Guard: the simulation path imports without the dashboard stack.

A backend that installs viva-fenics into its own environment (compose-api, an
HPC runner) gets no ``dashboard`` extra, so ``viva_superpowers`` is absent.
Importing the package there must neither fail nor silently drop the classes it
addresses as ``local:viva_fenics.processes.<module>.<Class>``. Each check runs in
a fresh interpreter so the blocked modules cannot leak from, or into, the rest
of the session.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# Always importable: their dolfinx import is deferred to update().
DOLFINX_FREE = ["FenicsReactiveTransportProcess", "FenicsThermophoreticTrapProcess"]

# Exported only where dolfinx is installed.
DOLFINX_BACKED = [
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
    "build_core",
    "register_processes",
]


def _run(blocked: list[str], names: list[str]) -> subprocess.CompletedProcess[str]:
    """Import viva_fenics with ``blocked`` modules made unimportable; check ``names``."""
    code = textwrap.dedent(
        f"""
        import sys
        for mod in {blocked!r}:
            sys.modules[mod] = None
        import viva_fenics
        import viva_fenics.processes
        missing = [n for n in {names!r} if not hasattr(viva_fenics, n)]
        assert not missing, f"viva_fenics silently dropped: {{missing}}"
        """
    )
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)


def test_imports_without_dashboard_package():
    result = _run(["viva_superpowers"], DOLFINX_FREE)
    assert result.returncode == 0, result.stderr


def test_dolfinx_classes_survive_without_dashboard_package():
    pytest.importorskip("dolfinx")
    result = _run(["viva_superpowers"], DOLFINX_FREE + DOLFINX_BACKED)
    assert result.returncode == 0, result.stderr


def _assert_import_failed(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode != 0, "import succeeded but should have raised"
    assert "ModuleNotFoundError" in result.stderr or "ImportError" in result.stderr, result.stderr
    assert "silently dropped" not in result.stderr, result.stderr


def test_unrelated_import_error_is_not_swallowed():
    """A missing runtime dependency must fail the import, not empty the package.

    Runs with or without dolfinx: viva_fenics.viz (the first guarded import)
    needs scipy before anything touches dolfinx.
    """
    _assert_import_failed(_run(["scipy"], DOLFINX_FREE))


def test_absent_fenicsx_stack_is_tolerated():
    """The one expected absence: no dolfinx leaves the dolfinx-free classes."""
    result = _run(["dolfinx"], DOLFINX_FREE)
    assert result.returncode == 0, result.stderr


def test_broken_dolfinx_is_not_swallowed():
    """dolfinx installed but failing to import (e.g. an API outside the pin) must raise."""
    pytest.importorskip("dolfinx")
    _assert_import_failed(_run(["dolfinx.fem.petsc"], DOLFINX_FREE + DOLFINX_BACKED))
