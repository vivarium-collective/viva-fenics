#!/usr/bin/env python3
"""Canonical run for the ``reaction-diffusion`` study -- now the investigation's
Turing-pattern showpiece (Gray-Scott, 2 diffusing species ⊕ 1 reaction, coupled
purely through shared bigraph stores).

Builds the ``turing_patterns`` composite -- TWO ``DiffusionProcess`` instances
(species U at Du, species V at Dv=Du/2) ⊕ ONE ``GrayScottReactionProcess``,
wired through shared ``stores.u`` / ``stores.v`` / ``stores.source_u`` /
``stores.source_v`` -- and runs it across THREE well-known Gray-Scott
parameter regimes (see ``REGIMES`` below) long enough for each to visibly
self-organize away from its near-uniform initial condition. Renders an
animated V-field for the baseline regime (the "pattern emerging from
uniformity" wow-shot) plus a final-state still for every regime, so the
morphology difference between regimes is directly visible.

Regime note (from development): the "classic" Gray-Scott spots regime
(F=0.035, k=0.065) was tried FIRST, with several seeding strategies
(localized patch only, patch + whole-field noise, patch sizes up to 0.3 of
the domain) -- all decayed straight back to the trivial (U=1, V=0) fixed
point within a few hundred ticks, i.e. genuinely SUBCRITICAL for this
composite's Du/Dv/dt/domain-size combination (see this study's report for
the calibration sweep: this is real reaction-diffusion physics, not a bug --
a perturbation below the local autocatalytic threshold decays exponentially
via the reaction's own linear term once diffusion dilutes it far enough).
The three regimes below were verified, during that same sweep, to grow
robustly and were used instead. ``turing_patterns``'s own default parameters
(F=0.037, k=0.06) are kept as this study's baseline -- they happen to sit in
the same "coral / dividing-spots" family as the classic "mitosis" regime
(F=0.0367, k=0.0649) from the Pearson (1993) atlas.

Standalone; run from the workspace root::

    pixi run python studies/reaction-diffusion/sims/run.py

Runtime note: the baseline (resolution=96, 6000 backward-Euler ticks at
dt=1.0, ~2 diffusion solves/tick) is the expensive part; the two variants run
at a coarser resolution (48) with fewer ticks, per this study's report on
feasible wall-time sizing. See that report for the measured wall-time of the
run that produced the committed viz/results.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import label
from process_bigraph import Composite

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[1]

from viva_fenics import fem, viz
from viva_fenics.core import build_core
from viva_fenics.composites.turing_patterns import turing_patterns
from vivarium_workbench.lib.run_log import append_run_event

SPEC_ID = "viva_fenics.composites.turing_patterns.turing_patterns"
STUDY_SLUG = "reaction-diffusion"
INVESTIGATION_SLUG = "fenics-showcase"

DU = 2e-5
DV = 1e-5
DT = 1.0

# Three Gray-Scott regimes (Pearson 1993 parameter atlas), verified during
# development to grow robustly under this composite's Du/Dv/dt/domain
# combination (see module docstring for why the classic "spots" combo was
# dropped). The baseline runs at a finer mesh/longer duration (it's the one
# with the animated wow-shot); the variants run cheaper, per this study's
# report on feasible wall-time sizing -- they still land clearly in
# different morphological regimes.
REGIMES = [
    {"name": "baseline", "label": "coral / mitosis-like", "F": 0.037, "k": 0.06,
     "resolution": 96, "n_steps": 6000, "animate": True},
    {"name": "labyrinth", "label": "labyrinth", "F": 0.03, "k": 0.06,
     "resolution": 48, "n_steps": 4000, "animate": False},
    {"name": "stripes", "label": "stripes / worms", "F": 0.055, "k": 0.062,
     "resolution": 48, "n_steps": 4000, "animate": False},
]

ANIMATION_FRAME_BUDGET = 30  # viz.field_animation_html guidance: <=~30 frames

# Behavior-test thresholds -- see study.yaml's expected_behavior /
# behavior_tests for the same numbers with rationale. Calibrated with
# headroom below what the shorter development-sweep runs (1500-2500 ticks,
# coarser mesh) already achieved; the production run below has more ticks
# and, for the baseline, a finer mesh, so should clear these comfortably --
# see this study's report for the ACHIEVED production values.
GROWTH_RATIO_MIN = 5.0          # pattern-emerges: var(V) growth over the baseline run (achieved 10.5x)
FIELD_BOUND = (-0.5, 1.5)        # pattern-bounded: loose [0,1]-ish band (see FAST tests)
COVERAGE_RANGE = (0.1, 0.7)      # pattern-bounded: fraction of domain "inside" a pattern feature (achieved 0.483)
COVERAGE_DIVERSITY_MIN = 0.1     # regimes-differ: spread of coverage across the 3 regimes (achieved 0.336)


def _pattern_metrics(coords, v_field, n_grid=100, blob_frac=0.5):
    """Connected-component ("spot"/domain) count + coverage fraction of a V
    field -- the structural signature of Turing pattern formation, unlike a
    bare variance (which can't distinguish "many small spots" from "one
    growing blob" or high-frequency noise). Interpolates the unstructured
    FEM field onto a regular grid (mirrors ``viz._interpolate_to_grid``),
    thresholds at ``blob_frac`` of the field's own [min, max] range, and
    labels connected components via ``scipy.ndimage.label`` (4-connectivity,
    the library default).

    Returns (n_blobs, coverage_fraction, vmin, vmax).
    """
    coords = np.asarray(coords, dtype=float)
    v_field = np.asarray(v_field, dtype=float)
    x, y = coords[:, 0], coords[:, 1]
    xi = np.linspace(float(x.min()), float(x.max()), n_grid)
    yi = np.linspace(float(y.min()), float(y.max()), n_grid)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata(coords, v_field, (XI, YI), method="linear")
    valid = np.isfinite(ZI)
    vmin = float(np.nanmin(ZI))
    vmax = float(np.nanmax(ZI))
    threshold = vmin + blob_frac * (vmax - vmin)
    mask = valid & (ZI > threshold)
    _labeled, n_blobs = label(mask)
    n_valid = int(valid.sum())
    coverage = float(mask.sum()) / n_valid if n_valid else 0.0
    return int(n_blobs), coverage, vmin, vmax


def _run_regime(core, cfg):
    resolution = cfg["resolution"]
    doc = turing_patterns(
        core, resolution=resolution, F=cfg["F"], k=cfg["k"], Du=DU, Dv=DV, dt=DT,
    )
    # Lightweight emitter: only the two scalar mass sensors. The heavy
    # u/v fields are snapshotted directly from the live composite state
    # (see below) -- accumulating full fields every tick in the RAMEmitter
    # would be gigabytes at this mesh resolution / tick count (mirrors
    # navier-stokes' run.py "_lightweight_vortex_street_doc" pattern).
    doc["emitter"]["config"]["emit"] = {"integral_u": "float", "integral_v": "float"}
    doc["emitter"]["inputs"] = {
        "integral_u": ["stores", "integral_u"],
        "integral_v": ["stores", "integral_v"],
    }
    sim = Composite({"state": doc}, core=core)

    _, V = fem.build_mesh("unit_square", resolution, degree=1)
    coords = fem.node_coords(V)
    v0 = np.array(doc["stores"]["v"], dtype=float)

    n_steps = cfg["n_steps"]
    frames_v, frame_times = [], []
    t0 = time.time()
    if cfg["animate"]:
        chunk = max(1, n_steps // ANIMATION_FRAME_BUDGET)
        n_chunks = n_steps // chunk
        for i in range(n_chunks):
            sim.run(chunk * DT)
            t_now = (i + 1) * chunk * DT
            frame_times.append(t_now)
            frames_v.append(np.array(sim.state["stores"]["v"], dtype=float).copy())
            print(
                f"[reaction-diffusion] {cfg['name']} t={t_now:.0f}/{n_steps * DT:.0f} "
                f"wall={time.time() - t0:.1f}s",
                flush=True,
            )
    else:
        sim.run(n_steps * DT)
    wall = time.time() - t0

    u_final = np.array(sim.state["stores"]["u"], dtype=float)
    v_final = np.array(sim.state["stores"]["v"], dtype=float)
    n_blobs, coverage, vmin, vmax = _pattern_metrics(coords, v_final)

    return {
        "coords": coords,
        "v0": v0,
        "u_final": u_final,
        "v_final": v_final,
        "frames_v": frames_v,
        "frame_times": frame_times,
        "wall": wall,
        "n_blobs": n_blobs,
        "coverage": coverage,
        "vmin": vmin,
        "vmax": vmax,
        "var0": float(np.var(v0)),
        "var_final": float(np.var(v_final)),
    }


def main() -> int:
    run_id = uuid.uuid4().hex
    started = time.time()
    n_steps_estimate = sum(r["n_steps"] for r in REGIMES)
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id,
        "event": "started",
        "spec_id": SPEC_ID,
        "label": STUDY_SLUG,
        "started_at": started,
        "status": "running",
        "n_steps": n_steps_estimate,
        "emitter": "ram",
        "origin": "canonical_run",
        "study_slug": STUDY_SLUG,
        "investigation_slug": INVESTIGATION_SLUG,
        "params": {"Du": DU, "Dv": DV, "dt": DT, "regimes": [
            {"name": r["name"], "F": r["F"], "k": r["k"],
             "resolution": r["resolution"], "n_steps": r["n_steps"]}
            for r in REGIMES
        ]},
    })

    try:
        core = build_core()
        results = {}
        for cfg in REGIMES:
            print(
                f"[reaction-diffusion] running {cfg['name']} ({cfg['label']}): "
                f"F={cfg['F']} k={cfg['k']} resolution={cfg['resolution']} "
                f"n_steps={cfg['n_steps']}",
                flush=True,
            )
            results[cfg["name"]] = _run_regime(core, cfg)
            r = results[cfg["name"]]
            print(
                f"[reaction-diffusion] {cfg['name']} done: wall={r['wall']:.1f}s "
                f"var0={r['var0']:.4g} var_final={r['var_final']:.4g} "
                f"n_blobs={r['n_blobs']} coverage={r['coverage']:.3f} "
                f"v_range=[{r['vmin']:.4f},{r['vmax']:.4f}]",
                flush=True,
            )

        baseline = results["baseline"]
        growth_ratio = baseline["var_final"] / max(baseline["var0"], 1e-12)
        pattern_emerges = growth_ratio >= GROWTH_RATIO_MIN

        all_bounded = all(
            np.all(r["u_final"] > FIELD_BOUND[0]) and np.all(r["u_final"] < FIELD_BOUND[1])
            and np.all(r["v_final"] > FIELD_BOUND[0]) and np.all(r["v_final"] < FIELD_BOUND[1])
            for r in results.values()
        )
        coverage_in_range = COVERAGE_RANGE[0] <= baseline["coverage"] <= COVERAGE_RANGE[1]
        pattern_structured = all_bounded and coverage_in_range

        coverages = [r["coverage"] for r in results.values()]
        coverage_spread = max(coverages) - min(coverages)
        regimes_differ = coverage_spread >= COVERAGE_DIVERSITY_MIN

        # --- viz ---
        viz_dir = STUDY_DIR / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)

        animation_html = viz.field_animation_html(
            baseline["coords"], baseline["frames_v"], baseline["frame_times"],
            f"Gray-Scott Turing pattern emerging (V field, {REGIMES[0]['label']}, "
            f"F={REGIMES[0]['F']} k={REGIMES[0]['k']}) -- 2x DiffusionProcess ⊕ GrayScottReactionProcess",
        )
        (viz_dir / "turing_pattern_animation.html").write_text(animation_html)

        for cfg in REGIMES:
            r = results[cfg["name"]]
            still_html = viz.field_heatmap_html(
                r["coords"], r["v_final"],
                f"{cfg['label']} (F={cfg['F']}, k={cfg['k']}) -- final V field "
                f"at t={cfg['n_steps'] * DT:.0f}",
            )
            (viz_dir / f"turing_pattern_{cfg['name']}_final.html").write_text(still_html)

        # remove the stale pre-rebuild animation file, if present, so the
        # study directory doesn't carry a dangling Fisher-KPP artifact
        # alongside the new Turing-pattern viz.
        stale = viz_dir / "reaction_diffusion_animation.html"
        if stale.exists():
            stale.unlink()
    except Exception:
        append_run_event(WORKSPACE_ROOT, {
            "run_id": run_id,
            "event": "completed",
            "completed_at": time.time(),
            "n_steps": 0,
            "status": "failed",
        })
        raise

    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id,
        "event": "completed",
        "completed_at": time.time(),
        "n_steps": n_steps_estimate,
        "status": "completed",
    })

    total_wall = time.time() - started
    print(
        f"[reaction-diffusion] total wall-time: {total_wall:.1f}s ({total_wall / 60:.1f} min) "
        f"across {len(REGIMES)} regimes"
    )
    for cfg in REGIMES:
        r = results[cfg["name"]]
        print(
            f"[reaction-diffusion] {cfg['name']} ({cfg['label']}): F={cfg['F']} k={cfg['k']} "
            f"resolution={cfg['resolution']} n_steps={cfg['n_steps']} wall={r['wall']:.1f}s "
            f"var0={r['var0']:.4g} var_final={r['var_final']:.4g} growth={r['var_final'] / max(r['var0'], 1e-12):.3g}x "
            f"n_blobs={r['n_blobs']} coverage={r['coverage']:.3f}"
        )
    print(
        f"[reaction-diffusion] baseline var(V) growth = {growth_ratio:.3g}x "
        f"-> {'PASS' if pattern_emerges else 'FAIL'} (pattern-emerges >= {GROWTH_RATIO_MIN}x)"
    )
    print(
        f"[reaction-diffusion] baseline coverage = {baseline['coverage']:.3f}, all fields bounded={all_bounded} "
        f"-> {'PASS' if pattern_structured else 'FAIL'} "
        f"(pattern-structured: coverage in {COVERAGE_RANGE}, fields in {FIELD_BOUND})"
    )
    print(
        f"[reaction-diffusion] coverage spread across regimes = {coverage_spread:.3f} "
        f"({dict((n, round(r['coverage'], 3)) for n, r in results.items())}) "
        f"-> {'PASS' if regimes_differ else 'FAIL'} (regimes-differ >= {COVERAGE_DIVERSITY_MIN})"
    )
    print(f"[reaction-diffusion] viz written: {viz_dir}")
    print(f"[reaction-diffusion] run recorded in {WORKSPACE_ROOT / '.pbg' / 'runs.jsonl'}")

    all_pass = pattern_emerges and pattern_structured and regimes_differ
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
