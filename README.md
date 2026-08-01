# viva-fenics

A [process-bigraph](https://github.com/vivarium-collective/process-bigraph) wrapper
for [FEniCSx](https://fenicsproject.org/) (`dolfinx`) — the real finite-element
solver, not a mock or a from-scratch reimplementation. It bridges dolfinx's mesh
generation, function spaces, and variational solvers into pbg Steps/Processes so
FEM models can be composed, run, and inspected through the
[vivarium-workbench](https://github.com/vivarium-collective/vivarium-workbench)
just like any other Vivarium simulator.

> **Wraps the real dolfinx.** `dolfinx` is installed via `conda-forge` (through
> `pixi`), not vendored or approximated — every study below solves an actual
> finite-element problem with dolfinx's assembler and PETSc-backed linear solves.

## Install

This workspace uses [pixi](https://pixi.sh) to manage the conda + PyPI
dependency stack (dolfinx's PETSc/MPI stack needs conda-forge; pbg packages come
from PyPI/git).

```bash
pixi install          # resolves + installs the full env (conda + pip layers)
pixi run pytest -q    # sanity-check: run the test suite
```

Everything below is invoked via `pixi run <command>` so it executes inside the
resolved environment.

## The 7 composite generators

Each generator builds one pbg composite wrapping a real dolfinx solve:

| Generator | Module | What it solves |
|---|---|---|
| `poisson_baseline` | `viva_fenics.composites.poisson` | Steady Poisson equation, manufactured (known-exact) solution |
| `mesh_convergence` | `viva_fenics.composites.convergence` | Poisson solved across a resolution sweep, for an O(h²) convergence check |
| `transient_diffusion` | `viva_fenics.composites.diffusion` | Time-stepped (backward-Euler) diffusion of an initial field |
| `reaction_diffusion` | `viva_fenics.composites.reaction_diffusion` | `DiffusionProcess` ⊕ `LogisticReactionProcess` coupled through shared bigraph stores |
| `navier_stokes` | `viva_fenics.composites.flow` | Lid-driven cavity flow, P2/P1 velocity-pressure, IPCS operator splitting |
| `moving_boundary` | `viva_fenics.composites.moving_boundary` | Diffusion on a prescribed-ALE deforming mesh, re-solved every substep |
| `complex_geometry` | `viva_fenics.composites.complex_geometry` | Poisson solved on gmsh-built non-rectangular domains (obstacle / L-shape / annulus) |

The shared dolfinx bridge (mesh/function-space/solve helpers) lives in
`viva_fenics/fem.py` and `viva_fenics/fem_gmsh.py`; process/Step wrappers live
under `viva_fenics/processes/`.

## The `fenics-showcase` investigation

`investigations/fenics-showcase/` is the seven-study arc that builds the case:
*a real external FEM solver can be wrapped as pbg Steps/Processes and composed
like any other pbg model.* Two validation studies establish numerical fidelity,
one extends it to time-stepping, one demonstrates genuine cross-process
composition (the investigation's headline result), and three advanced studies
push into fluids, deforming domains, and non-trivial geometry:

1. **`poisson-validation`** — manufactured-solution correctness: the FEM solution
   matches the known-exact quadratic solution to floating-point tolerance.
2. **`mesh-convergence`** — sweeps mesh resolution and confirms the L2 error
   converges at the expected O(h²) rate for P1 elements.
3. **`transient-diffusion`** — time-stepped diffusion of an initial gaussian bump;
   checks the peak decays and total mass is conserved under backward-Euler.
4. **`reaction-diffusion`** — couples an independently-authored `DiffusionProcess`
   and `LogisticReactionProcess` through shared bigraph stores; Fisher-KPP-like
   mass growth emerges purely from that wiring, with neither process aware of
   the other.
5. **`navier-stokes`** *(advanced)* — lid-driven cavity flow to quasi-steady
   state, checked for an approximately divergence-free velocity field and
   stability across a Reynolds-number sweep.
6. **`moving-boundary`** *(advanced)* — diffusion on a domain whose boundary
   follows a prescribed law (ALE mesh motion), checked against the analytic
   boundary trajectory.
7. **`complex-geometry`** *(advanced)* — Poisson solved on three gmsh-built
   non-rectangular domains (obstacle, L-shape, annulus), checked for a finite,
   bounded, non-trivial solution on each.

All seven currently pass their behavior tests (see each study's `study.yaml` /
the published dashboard for the exact criteria and results).

## Running a study

Each study has a bespoke `sims/run.py` runner that builds the composite, runs
it, evaluates the behavior tests, renders its Plotly viz to `viz/*.html`, and
records the run into the workspace's `.pbg/runs.jsonl` run log:

```bash
pixi run python studies/poisson-validation/sims/run.py
# ... one per study: mesh-convergence, transient-diffusion, reaction-diffusion,
#     navier-stokes, moving-boundary, complex-geometry
```

`.pbg/` is gitignored runtime state — re-run the studies locally to regenerate
it; nothing under `.pbg/` is committed.

## Publishing the read-only dashboard

The workspace ships a self-contained static-bundle publish flow
(`viva_superpowers.publish_assets.emit(...)`, scaffolded once) that produces:

- `scripts/publish_dashboard.sh` — builds the static bundle locally
  (`pixi run bash scripts/publish_dashboard.sh [out-dir]`).
- `.github/workflows/publish-dashboard.yml` — on every push to `main`, builds
  the same bundle in CI and pushes it to the `gh-pages` branch's `dashboard/`
  path.

To preview the bundle locally:

```bash
pixi run bash scripts/publish_dashboard.sh reports/published/dashboard
# then open reports/published/dashboard/index.html
```

Once deployed, the live read-only dashboard is served at
**https://vivarium-collective.github.io/viva-fenics/dashboard/** — a full
mirror of the `fenics-showcase` investigation and its seven studies, browsable
with no server, including each study's interactive Plotly/three.js
visualization embedded read-only.

## Layout

```
viva_fenics/
├── fem.py, fem_gmsh.py       # dolfinx bridge (mesh, function spaces, solves)
├── processes/                 # pbg Step/Process wrappers
└── composites/                 # the 7 generators listed above
investigations/fenics-showcase/ # the showcase investigation
studies/<slug>/
├── study.yaml                  # question/hypothesis/behavior_tests/visualizations
├── sims/run.py                 # bespoke runner: run → evaluate → render → log
└── viz/*.html                  # rendered interactive Plotly/three.js output
```

## Notes

- **Not a mock.** Every composite here bridges `dolfinx`'s real assembler and
  linear solves — no reimplemented numerics, no stub returning canned data.
- **Platform.** `pixi.toml` currently pins `osx-arm64`; dolfinx's conda-forge
  build is the long pole for other platforms.
