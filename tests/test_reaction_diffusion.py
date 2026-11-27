"""Tests for FenicsxReactionDiffusionProcess (multi-species structured-grid RD).

The grid-helper, interface and dolfinx-free import checks always run (the module
keeps its dolfinx import lazy, like spatial_transport). The actual FEM solve needs
the conda dolfinx stack and is skipped otherwise — the end-to-end hybrid coupling is
exercised in viva-pde-particle, which reuses this process.
"""
import importlib.util

import numpy as np
import pytest

from viva_fenics.processes.reaction_diffusion import (
    FenicsxReactionDiffusionProcess,
    _StructuredGrid,
    reaction_rates,
)

_HAVE_DOLFINX = importlib.util.find_spec("dolfinx") is not None


def test_importable_without_dolfinx():
    # The process + its module import with no dolfinx present (import is lazy).
    import viva_fenics
    assert hasattr(viva_fenics, "FenicsxReactionDiffusionProcess")


def test_interface():
    assert set(FenicsxReactionDiffusionProcess.config_schema) == {"grid", "pde", "dt", "mass"}
    inst = FenicsxReactionDiffusionProcess.__new__(FenicsxReactionDiffusionProcess)
    assert inst.inputs() == {"fields": "map[array[float]]", "external_conc": "map[array[float]]"}
    assert inst.outputs() == {"fields": "map[overwrite[array[float]]]"}


def test_structured_grid_matches_conventions():
    g = _StructuredGrid({"origin": [0.0, 0.0], "size": [10.0, 10.0], "num": [11, 11]})
    assert g.dim == 2
    assert g.shape == (11, 11)          # (Nz, Ny, Nx) restricted to dim
    assert g.num_elements == 121
    # nearest-node mapping: a point near the far corner lands on index (10, 10)
    (iz, ix) = g.node_index(np.array([[10.0, 10.0]]))
    assert int(iz[0]) == 10 and int(ix[0]) == 10
    with pytest.raises(ValueError):
        _StructuredGrid({"origin": [0.0], "size": [1.0, 1.0], "num": [2, 2]})


def test_reaction_rates_mass_action():
    shape = (2, 2)
    terms = [{"species": "C", "coeff": 1.0, "k": 2.0, "reactants": {"A": 1, "B": 1}}]
    conc = {"A": np.full(shape, 3.0), "B": np.full(shape, 4.0), "C": np.zeros(shape)}
    rates = reaction_rates(terms, ["C"], conc, shape)
    assert np.allclose(rates["C"], 2.0 * 3.0 * 4.0)  # k*coeff*A*B


def test_fail_loud_on_underdeclared_external_species():
    # An EXPLICIT external_species list that misses a referenced species fails loud
    # (an empty/omitted list auto-derives every non-species reactant instead).
    with pytest.raises(ValueError, match="species not in species/external_species"):
        FenicsxReactionDiffusionProcess({
            "grid": {"origin": [0.0, 0.0], "size": [1.0, 1.0], "num": [2, 2]},
            "pde": {"species": {"A": {"diffusion": 1.0}},
                    "terms": [{"species": "A", "coeff": -1.0, "k": 1.0, "reactants": {"A": 1, "Z": 1}}],
                    "external_species": ["B"]},  # explicit, but misses Z
        })


@pytest.mark.skipif(not _HAVE_DOLFINX, reason="dolfinx (conda FEniCSx stack) not installed")
def test_pure_diffusion_conserves_mass():
    """One species, no reactions: a diffusion step conserves total mass (lumped mass)."""
    grid = {"origin": [0.0, 0.0], "size": [10.0, 10.0], "num": [11, 11]}
    proc = FenicsxReactionDiffusionProcess({
        "grid": grid,
        "pde": {"species": {"A": {"diffusion": 1.0}}, "terms": [], "external_species": []},
        "dt": 0.05,
    })
    f0 = np.zeros((11, 11)); f0[5, 5] = 100.0
    out = proc.update({"fields": {"A": f0}}, 0.05)
    assert out["fields"]["A"].sum() == pytest.approx(f0.sum(), rel=1e-6)
