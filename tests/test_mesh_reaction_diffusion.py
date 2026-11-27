"""Tests for FenicsxMeshReactionDiffusionProcess (unstructured P1 mesh RD).

Interface, the build_space seam, and the fail-loud guard run anywhere (dolfinx import
is lazy). The gmsh sphere build + solve need the conda FEniCSx stack and are skipped
otherwise — the hybrid coupling is exercised in viva-pde-particle, which subclasses this
and overrides build_space with its own geometry framework.
"""
import importlib.util

import numpy as np
import pytest

from viva_fenics.processes.mesh_reaction_diffusion import (
    FenicsxMeshReactionDiffusionProcess,
    MeshSpace,
)

_HAVE_DOLFINX = importlib.util.find_spec("dolfinx") is not None


def test_importable_without_dolfinx():
    import viva_fenics
    assert hasattr(viva_fenics, "FenicsxMeshReactionDiffusionProcess")


def test_interface():
    assert set(FenicsxMeshReactionDiffusionProcess.config_schema) == {"mesh", "pde", "dt"}
    inst = FenicsxMeshReactionDiffusionProcess.__new__(FenicsxMeshReactionDiffusionProcess)
    assert inst.inputs() == {"field_dofs": "map[array[float]]", "external_conc": "map[array[float]]"}
    assert inst.outputs() == {"field_dofs": "map[overwrite[array[float]]]"}


def test_base_build_space_rejects_non_sphere():
    # The kind check runs before any dolfinx import, so this works without the stack.
    inst = FenicsxMeshReactionDiffusionProcess.__new__(FenicsxMeshReactionDiffusionProcess)
    with pytest.raises(ValueError, match="override build_space"):
        inst.build_space({"kind": "geometry"})


def test_build_space_is_an_overridable_seam():
    # A consumer plugs in its own mesh/geometry by overriding build_space; the RD engine
    # (initialize/update) is inherited unchanged. We verify the seam dispatches to the
    # override (full run needs dolfinx and is covered below / in viva-pde-particle).
    class _Custom(FenicsxMeshReactionDiffusionProcess):
        def build_space(self, mesh_config):
            return MeshSpace(msh=None, V="my-space", ml=np.ones(3))
    inst = _Custom.__new__(_Custom)
    space = inst.build_space({"kind": "whatever"})
    assert space.V == "my-space" and list(space.ml) == [1.0, 1.0, 1.0]


def test_fail_loud_on_underdeclared_external_species():
    with pytest.raises(ValueError, match="species not in species/external_species"):
        FenicsxMeshReactionDiffusionProcess({
            "mesh": {"kind": "sphere", "radius": 1.0},
            "pde": {"species": {"A": {"diffusion": 1.0}},
                    "terms": [{"species": "A", "coeff": -1.0, "k": 1.0, "reactants": {"A": 1, "Z": 1}}],
                    "external_species": ["B"]},  # explicit, misses Z -> raises before build_space
        })


@pytest.mark.skipif(not _HAVE_DOLFINX, reason="dolfinx (conda FEniCSx stack) not installed")
def test_sphere_diffusion_conserves_mass():
    proc = FenicsxMeshReactionDiffusionProcess({
        "mesh": {"kind": "sphere", "radius": 1.0, "h": 0.3},
        "pde": {"species": {"A": {"diffusion": 1.0}}, "terms": [], "external_species": []},
        "dt": 0.05,
    })
    u0 = proc.initial_dofs({"A": 0.0})
    u0["A"][0] = 100.0
    m0 = float((proc.ml * u0["A"]).sum())
    out = proc.update({"field_dofs": u0}, 0.05)
    assert float((proc.ml * out["field_dofs"]["A"]).sum()) == pytest.approx(m0, rel=1e-6)
