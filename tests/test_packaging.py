"""Guard: every third-party module the package imports is declared.

A backend installs viva-fenics from git without extras into an environment that
already has the conda FEniCSx stack. Each import therefore has to come either
from ``[project].dependencies`` (PyPI) or from the conda stack in ``pixi.toml``;
anything else works locally only because some other package pulled it in.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "viva_fenics"

# import name -> PyPI distribution declared in pyproject.toml
PYPI = {
    "numpy": "numpy",
    "scipy": "scipy",
    "plotly": "plotly",
    "process_bigraph": "process-bigraph",
    "bigraph_schema": "bigraph-schema",
    "yaml": "pyyaml",
}

# import name -> conda package in pixi.toml that provides it (ufl, basix,
# mpi4py and petsc4py are dependencies of fenics-dolfinx).
CONDA = {
    "dolfinx": "fenics-dolfinx",
    "ufl": "fenics-dolfinx",
    "basix": "fenics-dolfinx",
    "mpi4py": "fenics-dolfinx",
    "petsc4py": "fenics-dolfinx",
    "gmsh": "python-gmsh",
}


def _requirement_name(spec: str) -> str:
    return re.split(r"[\s\[<>=!~;@]", spec, maxsplit=1)[0].lower()


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def _runtime_requirements() -> set[str]:
    return {_requirement_name(spec) for spec in _pyproject()["project"]["dependencies"]}


def _imported_top_level_modules() -> set[str]:
    found: set[str] = set()
    for path in PACKAGE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found - set(sys.stdlib_module_names) - {"viva_fenics", "__future__"}


def test_scipy_is_a_runtime_dependency():
    assert "scipy" in _runtime_requirements()


def test_every_import_is_declared():
    imported = _imported_top_level_modules()
    unknown = imported - PYPI.keys() - CONDA.keys()
    assert not unknown, f"undeclared third-party imports in viva_fenics: {sorted(unknown)}"

    runtime = _runtime_requirements()
    missing_pypi = sorted(PYPI[m] for m in imported & PYPI.keys() if PYPI[m] not in runtime)
    assert not missing_pypi, f"imported but not in [project].dependencies: {missing_pypi}"

    pixi_deps = tomllib.loads((ROOT / "pixi.toml").read_text())["dependencies"]
    missing_conda = sorted({CONDA[m] for m in imported & CONDA.keys()} - pixi_deps.keys())
    assert not missing_conda, f"imported but not in pixi.toml [dependencies]: {missing_conda}"


def test_dashboard_stack_is_an_extra_not_a_runtime_dependency():
    project = _pyproject()["project"]
    runtime = _runtime_requirements()
    assert "viva-superpowers" not in runtime
    assert "vivarium-workbench" not in runtime
    dashboard = {_requirement_name(s) for s in project["optional-dependencies"]["dashboard"]}
    assert {"viva-superpowers", "vivarium-workbench"} <= dashboard
