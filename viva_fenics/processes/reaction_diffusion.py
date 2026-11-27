"""Multi-species reaction-diffusion on a structured grid (FEniCSx Q1 + scipy stepping).

The counterpart to :mod:`viva_fenics.processes.diffusion` for **hybrid PDE/particle**
co-simulation: unlike ``DiffusionProcess`` (single species, built-in unit-square mesh),
this solves an arbitrary set of species with mass-action reaction terms on a
structured grid given as ``{origin, size, num}`` — the same grid a coupled particle
engine (e.g. ``viva_smoldyn.SmoldynHybridProcess``) bins onto, so fields and particle
histograms line up node for node. External species (read, not evolved — e.g. particle
species) enter the reaction rates through the ``external_conc`` port.

dolfinx assembles the Q1 stiffness/mass operators once (``q1_operators``); the per-step
solve is scipy-sparse. The dolfinx import is lazy, so the module loads without it; a
build is only needed to construct the operators.

Extracted from viva-pde-particle's vendored ``FenicsxReactionDiffusion`` so workspaces
reuse it instead of importing dolfinx directly. The ``grid`` dict shape + validation is
identical to ``viva_smoldyn.SmoldynHybridProcess`` on purpose.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from process_bigraph import Process


class _StructuredGrid:
    """Node-centred Cartesian grid from a plain ``{origin, size, num}`` config.

    N nodes per axis (spacing L/(N-1)); a point maps to the nearest node. Arrays have
    shape ``(Nz, Ny, Nx)`` restricted to the model dimension (C order = the layout the
    coupled particle engine uses). The validation matches the Smoldyn side byte-for-byte.
    """

    def __init__(self, config: dict):
        self.origin = [float(v) for v in config["origin"]]
        self.size = [float(v) for v in config["size"]]
        self.num = [int(v) for v in config["num"]]
        if not (len(self.origin) == len(self.size) == len(self.num)) or not 1 <= len(self.num) <= 3:
            raise ValueError("grid origin, size and num must have equal length 1, 2 or 3")
        if any(n < 1 for n in self.num) or any(s <= 0 for s in self.size):
            raise ValueError("grid num must be >= 1 and size > 0")

    @property
    def dim(self) -> int:
        return len(self.num)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(reversed(self.num))

    @property
    def num_elements(self) -> int:
        return int(np.prod(self.num))

    def node_index(self, points: np.ndarray) -> tuple[np.ndarray, ...]:
        """Nearest-node array indices (z, y, x order) for points of shape (n, dim), clamped."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        idx = []
        for d in range(self.dim):
            n = self.num[d]
            if n == 1:
                i = np.zeros(len(points), dtype=int)
            else:
                i = ((points[:, d] - self.origin[d]) * (n - 1) / self.size[d] + 0.5).astype(int)
                i = np.clip(i, 0, n - 1)
            idx.append(i)
        return tuple(reversed(idx))


def q1_operators(grid: _StructuredGrid):
    """(K, M, dof_to_node): Q1 stiffness and mass (scipy CSR, DOF order) and each DOF's
    flat grid index. dolfinx builds the structured mesh + assembles; import is lazy."""
    import ufl
    from dolfinx import fem, mesh
    from mpi4py import MPI

    lo = np.array(grid.origin)
    hi = lo + np.array(grid.size)
    cells = [n - 1 for n in grid.num]
    if grid.dim == 2:
        msh = mesh.create_rectangle(MPI.COMM_SELF, [lo, hi], cells, mesh.CellType.quadrilateral)
    elif grid.dim == 3:
        msh = mesh.create_box(MPI.COMM_SELF, [lo, hi], cells, mesh.CellType.hexahedron)
    else:
        msh = mesh.create_interval(MPI.COMM_SELF, cells[0], [lo[0], hi[0]])
    V = fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    K = fem.assemble_matrix(fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)).to_scipy()
    M = fem.assemble_matrix(fem.form(u * v * ufl.dx)).to_scipy()
    X = V.tabulate_dof_coordinates()[:, : grid.dim]
    idx = grid.node_index(X)  # (z, y, x) array indices
    dof_to_node = np.ravel_multi_index(idx, grid.shape)
    if len(np.unique(dof_to_node)) != grid.num_elements:
        raise RuntimeError("Q1 DOFs do not map one-to-one onto grid nodes")
    return K.tocsr(), M.tocsr(), dof_to_node


def reaction_rates(terms, species, conc, shape):
    """dC/dt from mass-action terms (uM/s) for each continuous species.

    Each term is ``{species, coeff, k, reactants: {name: order}}``; shared, engine-neutral
    (no dolfinx) so the same model drives this and any other PDE engine.
    """
    rates = {s: np.zeros(shape) for s in species}
    for term in terms:
        r = np.full(shape, term["coeff"] * term["k"])
        for s, n in term["reactants"].items():
            r = r * conc[s] ** n
        rates[term["species"]] += r
    return rates


class FenicsxReactionDiffusionProcess(Process):
    """Multi-species reaction-diffusion on a structured grid (FEniCSx Q1, scipy stepping).

    Config:
        grid: ``{origin, size, num}`` — the shared structured grid (identical shape/validation
            to the coupled particle engine).
        pde: ``{species: {name: {diffusion}}, terms: [...], external_species?: [...]}`` —
            per-species diffusion + mass-action reaction terms. ``external_species`` are read
            (via ``external_conc``) but not evolved here; defaulted to every reactant not in
            this engine's species.
        dt: time step (s).
        mass: ``"lumped"`` (default) or ``"consistent"`` mass matrix.
    """

    config_schema = {
        "grid": "map",
        "pde": "map",
        "dt": {"_type": "float", "_default": 0.01},
        "mass": {"_type": "string", "_default": "lumped"},
    }

    def initialize(self, config):
        if config["mass"] not in ("lumped", "consistent"):
            raise ValueError(f"mass must be 'lumped' or 'consistent', got {config['mass']!r}")
        self.grid = _StructuredGrid(config["grid"])
        self.pde = config["pde"]
        self.species = list(self.pde["species"])
        self.external_species = list(self.pde.get("external_species") or sorted(
            {s for t in self.pde["terms"] for s in t["reactants"]} - set(self.species)))
        # Fail loud if a reaction term references a species we neither evolve nor read
        # (the analog of the Smoldyn side's requiredFields() vs field_species guard).
        unknown = {s for t in self.pde["terms"] for s in t["reactants"]} - set(self.species) - set(self.external_species)
        if unknown:
            raise ValueError(
                f"reaction terms reference species not in species/external_species: {sorted(unknown)}")
        self.dt = float(config["dt"])
        K, M, self._dof_to_node = q1_operators(self.grid)
        if config["mass"] == "lumped":
            M = sp.diags(np.asarray(M.sum(axis=1)).ravel())
        self._M = M.tocsr()
        self._solvers = {
            name: spla.factorized((self._M + float(spec["diffusion"]) * self.dt * K).tocsc())
            for name, spec in self.pde["species"].items()
        }

    def inputs(self):
        return {"fields": "map[array[float]]", "external_conc": "map[array[float]]"}

    def outputs(self):
        return {"fields": "map[overwrite[array[float]]]"}

    def _to_dofs(self, arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr, dtype=float).ravel()[self._dof_to_node]

    def _to_grid(self, dofs: np.ndarray) -> np.ndarray:
        out = np.empty(self.grid.num_elements)
        out[self._dof_to_node] = dofs
        return out.reshape(self.grid.shape)

    def step(self, fields, external_conc):
        rates = reaction_rates(self.pde["terms"], self.species, {**external_conc, **fields}, self.grid.shape)
        return {
            s: self._to_grid(self._solvers[s](self._M @ (self._to_dofs(fields[s]) + self.dt * self._to_dofs(rates[s]))))
            for s in self.species
        }

    def update(self, state, interval):
        fields = {s: np.asarray(state["fields"][s], dtype=float) for s in self.species}
        ext = state.get("external_conc") or {}
        external_conc = {
            s: np.asarray(ext[s], dtype=float) if s in ext else np.zeros(self.grid.shape)
            for s in self.external_species
        }
        for _ in range(max(1, int(round(interval / self.dt)))):
            fields = self.step(fields, external_conc)
        return {"fields": fields}
