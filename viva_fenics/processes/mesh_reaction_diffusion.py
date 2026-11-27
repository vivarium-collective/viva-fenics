"""Multi-species reaction-diffusion on an unstructured P1 mesh (FEniCSx).

The unstructured-mesh counterpart to :class:`FenicsxReactionDiffusionProcess` (which is
on a structured grid): a general PDE engine that knows no grid and no particles. The PDE
is P1 on an unstructured mesh (e.g. a gmsh ball), lumped mass, backward-Euler diffusion +
explicit mass-action reactions; the authoritative state is the DOF vector ``field_dofs``.

The mesh/space is produced by :meth:`build_space`, a seam a consumer overrides to plug in
its own geometry framework (custom domains, grid<->mesh coupling) while reusing the RD
assembly + stepping here. The base builds a gmsh ball (``{kind: "sphere", radius, center,
h}``). dolfinx is imported lazily, so the module/class load + register in a dolfinx-free
uv venv; only running needs the conda FEniCSx stack.

Extracted from viva-pde-particle's vendored ``FenicsxMeshReactionDiffusion`` as the
general-mesh half of its FEniCS reuse — it subclasses this and overrides ``build_space``.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from process_bigraph import Process


def sphere_mesh(radius: float, center=(0.0, 0.0, 0.0), h: float | None = None):
    """Unstructured tetrahedral mesh of a ball (gmsh), as a dolfinx Mesh on COMM_SELF."""
    import gmsh
    from dolfinx.io import gmsh as gmshio
    from mpi4py import MPI

    h = h or radius / 6
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("ball")
        tag = gmsh.model.occ.addSphere(*center, radius)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [tag], 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h / 2)
        gmsh.model.mesh.generate(3)
        data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF, 0, gdim=3)
    finally:
        gmsh.finalize()
    return data.mesh if hasattr(data, "mesh") else data[0]


class MeshSpace:
    """A mesh with its P1 space and lumped mass — everything the RD engine needs, no grid."""

    __slots__ = ("msh", "V", "ml")

    def __init__(self, msh, V, ml):
        self.msh = msh
        self.V = V
        self.ml = np.asarray(ml).ravel()


class FenicsxMeshReactionDiffusionProcess(Process):
    """Reaction-diffusion on an unstructured P1 mesh.

    Config:
        mesh: mesh spec for :meth:`build_space`; the base supports
            ``{"kind": "sphere", "radius", "center", "h"}``.
        pde: ``{species: {name: {diffusion}}, terms: [...], external_species?: [...]}``.
        dt: time step (s).

    Override :meth:`build_space` to supply a custom mesh/geometry (returning anything with
    ``.V`` (a dolfinx P1 FunctionSpace) and ``.ml`` (lumped mass array)); the RD assembly +
    stepping here are reused unchanged.
    """

    config_schema = {
        "mesh": "map",
        "pde": "map",
        "dt": {"_type": "float", "_default": 0.01},
    }

    def build_space(self, mesh_config):
        """P1 space + lumped mass for the mesh. Base: a gmsh ball. Override for custom geometry."""
        if mesh_config.get("kind") != "sphere":
            raise ValueError(
                f"base build_space supports kind='sphere', got {mesh_config.get('kind')!r} "
                "— override build_space to supply a custom mesh/geometry")
        import ufl
        from dolfinx import fem

        msh = sphere_mesh(float(mesh_config["radius"]),
                          tuple(mesh_config.get("center", (0.0, 0.0, 0.0))),
                          mesh_config.get("h"))
        V = fem.functionspace(msh, ("Lagrange", 1))
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        M = fem.assemble_matrix(fem.form(u * v * ufl.dx)).to_scipy()
        return MeshSpace(msh, V, np.asarray(M.sum(axis=1)).ravel())

    def initialize(self, config):
        self.pde = config["pde"]
        self.species = list(self.pde["species"])
        self.external_species = list(self.pde.get("external_species") or sorted(
            {s for t in self.pde["terms"] for s in t["reactants"]} - set(self.species)))
        unknown = {s for t in self.pde["terms"] for s in t["reactants"]} - set(self.species) - set(self.external_species)
        if unknown:
            raise ValueError(
                f"reaction terms reference species not in species/external_species: {sorted(unknown)}")
        self.dt = float(config["dt"])
        space = self.build_space(config["mesh"])
        import ufl
        from dolfinx import fem

        self.V = space.V
        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        K = fem.assemble_matrix(fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)).to_scipy().tocsr()
        self.ml = np.asarray(space.ml).ravel()  # lumped mass = nodal control volumes (um^3)
        self.n_dofs = len(self.ml)
        self._solvers = {
            name: spla.factorized((sp.diags(self.ml) + float(spec["diffusion"]) * self.dt * K).tocsc())
            for name, spec in self.pde["species"].items()
        }

    def initial_dofs(self, initial: dict) -> dict:
        return {s: np.full(self.n_dofs, float(initial.get(s, 0.0))) for s in self.species}

    def inputs(self):
        return {"field_dofs": "map[array[float]]", "external_conc": "map[array[float]]"}

    def outputs(self):
        return {"field_dofs": "map[overwrite[array[float]]]"}

    def update(self, state, interval):
        from viva_fenics.processes.reaction_diffusion import reaction_rates

        u = {s: np.asarray(state["field_dofs"][s], dtype=float) for s in self.species}
        ext = state.get("external_conc") or {}
        pc = {s: np.asarray(ext[s], dtype=float) if s in ext else np.zeros(self.n_dofs)
              for s in self.external_species}
        for _ in range(max(1, int(round(interval / self.dt)))):
            rates = reaction_rates(self.pde["terms"], self.species, {**pc, **u}, (self.n_dofs,))
            u = {s: self._solvers[s](self.ml * (u[s] + self.dt * rates[s])) for s in self.species}
        return {"field_dofs": u}
