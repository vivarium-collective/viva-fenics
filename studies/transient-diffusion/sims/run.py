#!/usr/bin/env python3
"""Canonical run for the ``transient-diffusion`` study -- now the morphogen
gradient (Source-Diffusion-Degradation / "French flag") showpiece.

Builds the ``morphogen_gradient`` composite -- ``DiffusionProcess`` (with a
real Dirichlet source boundary c=c0 at x=0) ⊕ ``LinearDegradationProcess``
(``-k*c``), coupled purely through shared bigraph stores ``stores.solution``
/ ``stores.source`` -- and runs it to steady state across THREE decay-length
regimes (see ``REGIMES`` below: D fixed, k swept) so lambda=sqrt(D/k)'s
control over the gradient's spatial range is directly visible. Renders (a)
an animated field of the baseline gradient FORMING (spreading from the
source boundary to its steady exponential shape), (b) the three regimes'
steady 1D profiles overlaid with their analytic exp(-x/lambda) fits, and (c)
a French-flag positional-information readout (3 fate regions) per regime.

Governing PDE (emergent from the two independently-authored processes, not
implemented anywhere as one equation): dc/dt = D*laplacian(c) - k*c, with
c(x=0)=c0 (Dirichlet source boundary) and no-flux (natural Neumann)
elsewhere. Steady state: c(x) = c0*exp(-x/lambda), lambda = sqrt(D/k).

Fit-region note (from development): the domain is a UNIT square with a
no-flux far boundary (x=1), not a true semi-infinite domain -- the Neumann
condition forces dc/dx=0 at x=1, which biases the field upward relative to
the pure exponential near that edge (confirmed empirically: fitting the full
[0,1] range overstates lambda by ~20% for the baseline regime). Both the
decay-length fit and the French-flag threshold comparison below therefore
use only x in [2h, 0.4] (a couple of mesh cells past the source boundary,
comfortably short of the far-boundary influence) -- see this study's report
for the full sweep showing this region matches the analytic curve to <2%
for every regime tested, while the excluded far region visibly deviates.

Standalone; run from the workspace root::

    pixi run python studies/transient-diffusion/sims/run.py
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np
from process_bigraph import Composite

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[1]

from viva_fenics import fem, viz
from viva_fenics.core import build_core
from viva_fenics.composites.morphogen_gradient import morphogen_gradient
from vivarium_workbench.lib.run_log import append_run_event

SPEC_ID = "viva_fenics.composites.morphogen_gradient.morphogen_gradient"
STUDY_SLUG = "transient-diffusion"
INVESTIGATION_SLUG = "fenics-showcase"

D = 0.1
C0 = 1.0
RESOLUTION = 64
DT = 0.02
RUN_TIME = 16.0  # 800 backward-Euler ticks; verified (see report) to reach
# steady state (fitted lambda stable to 4 significant figures) by t~8 for
# the slowest (largest-lambda) regime below -- 16.0 leaves ~2x headroom.

# Three decay-length regimes: D fixed, k swept -- lambda=sqrt(D/k) shrinks
# as k grows, directly demonstrating lambda's control over the gradient's
# spatial range. k=0.5 (a longer-range regime, lambda~0.45) was tried during
# development and dropped: its lambda approaches the domain size closely
# enough that the far no-flux boundary's influence reaches into the fit
# region, degrading the decay-length fit to ~7% error -- an honest numerical
# limit of a unit-square domain, not a bug. The three regimes below all keep
# lambda comfortably smaller than the domain (<1/3), where boundary
# influence is negligible in the [2h, 0.4] fit region (see report).
REGIMES = [
    {"name": "baseline", "label": "baseline (long-range)", "k": 1.0, "animate": True},
    {"name": "short", "label": "short-range", "k": 2.0, "animate": False},
    {"name": "shorter", "label": "shorter-range", "k": 4.0, "animate": False},
]

ANIMATION_FRAME_BUDGET = 24  # viz.field_animation_html guidance: <=~30 frames

# French-flag positional-information thresholds (fraction of c0).
THETA_HIGH = 0.5 * C0
THETA_LOW = 0.1 * C0
REGION_LABELS = ["A (c > 0.5 c0)", "B (0.1-0.5 c0)", "C (c < 0.1 c0)"]

# Fit region for the decay-length + threshold-crossing measurements: past a
# couple of mesh cells from the source boundary, short of the far no-flux
# boundary's influence (see module docstring).
FIT_X_MIN_CELLS = 2.0
FIT_X_MAX = 0.4

# Behavior-test thresholds -- see study.yaml's expected_behavior /
# behavior_tests for the same numbers with rationale + achieved values.
LAMBDA_REL_ERR_MAX = 0.10       # exponential-gradient: |fit - analytic| / analytic
X_HIGH_REL_ERR_MAX = 0.10       # french-flag-thresholds: measured vs analytic x_high crossing
BOUNDARY_SHIFT_MIN = 0.05       # french-flag-thresholds: x_high(baseline) - x_high(shorter) >= this


def _fit_decay_length(x, c, resolution):
    """Least-squares fit of ln(c) vs x over the [2h, 0.4] window; returns
    (fitted_lambda, slope, n_points_used)."""
    x_min = FIT_X_MIN_CELLS / resolution
    mask = (x > x_min) & (x < FIT_X_MAX) & (c > 1e-9)
    slope, _intercept = np.polyfit(x[mask], np.log(c[mask]), 1)
    return -1.0 / slope, slope, int(mask.sum())


def _collapse_by_x(x, c):
    """Average c over nodes sharing the same x (the field is translationally
    symmetric in y since both the BC and initial condition depend only on
    x), giving a clean 1D profile for fitting/plotting/crossing-detection."""
    xs_unique = np.unique(np.round(x, 6))
    c_by_x = np.array([c[np.isclose(x, xv, atol=1e-6)].mean() for xv in xs_unique])
    order = np.argsort(xs_unique)
    return xs_unique[order], c_by_x[order]


def _threshold_crossing_x(x_sorted, c_sorted, theta):
    """First x (linearly interpolated) where the (monotonically decreasing)
    profile crosses below `theta`; None if it never does."""
    below = c_sorted < theta
    if not below.any():
        return None
    idx = int(np.argmax(below))
    if idx == 0:
        return float(x_sorted[0])
    x0, x1 = x_sorted[idx - 1], x_sorted[idx]
    c0_, c1 = c_sorted[idx - 1], c_sorted[idx]
    frac = (theta - c0_) / (c1 - c0_)
    return float(x0 + frac * (x1 - x0))


def _run_regime(core, cfg):
    k = cfg["k"]
    lam = float(np.sqrt(D / k))
    doc = morphogen_gradient(core, D=D, k=k, c0=C0, resolution=RESOLUTION, dt=DT)
    sim = Composite({"state": doc}, core=core)

    _, V = fem.build_mesh("unit_square", RESOLUTION, degree=1)
    coords = fem.node_coords(V)

    n_steps = round(RUN_TIME / DT)
    frames, frame_times = [], []
    t0 = time.time()
    if cfg["animate"]:
        chunk = max(1, n_steps // ANIMATION_FRAME_BUDGET)
        n_chunks = n_steps // chunk
        for i in range(n_chunks):
            sim.run(chunk * DT)
            t_now = (i + 1) * chunk * DT
            frame_times.append(t_now)
            frames.append(np.array(sim.state["stores"]["solution"], dtype=float).copy())
            print(
                f"[transient-diffusion] {cfg['name']} t={t_now:.2f}/{RUN_TIME:.1f} "
                f"wall={time.time() - t0:.1f}s",
                flush=True,
            )
    else:
        sim.run(RUN_TIME)
    wall = time.time() - t0

    c_final = np.array(sim.state["stores"]["solution"], dtype=float)
    x_sorted, c_sorted = _collapse_by_x(coords[:, 0], c_final)

    fitted_lambda, slope, n_fit = _fit_decay_length(x_sorted, c_sorted, RESOLUTION)

    x_high_analytic = -lam * np.log(THETA_HIGH / C0)
    x_low_analytic = -lam * np.log(THETA_LOW / C0)
    x_high_measured = _threshold_crossing_x(x_sorted, c_sorted, THETA_HIGH)
    x_low_measured = _threshold_crossing_x(x_sorted, c_sorted, THETA_LOW)

    return {
        "k": k,
        "lambda_analytic": lam,
        "lambda_fit": float(fitted_lambda),
        "lambda_rel_err": abs(fitted_lambda - lam) / lam,
        "n_fit": n_fit,
        "coords": coords,
        "c_final": c_final,
        "x_sorted": x_sorted,
        "c_sorted": c_sorted,
        "frames": frames,
        "frame_times": frame_times,
        "wall": wall,
        "x_high_analytic": x_high_analytic,
        "x_low_analytic": x_low_analytic,
        "x_high_measured": x_high_measured,
        "x_low_measured": x_low_measured,
        "c_min": float(c_final.min()),
        "c_max": float(c_final.max()),
    }


def main() -> int:
    run_id = uuid.uuid4().hex
    started = time.time()
    n_steps_estimate = round(RUN_TIME / DT) * len(REGIMES)
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
        "params": {"D": D, "c0": C0, "resolution": RESOLUTION, "dt": DT, "run_time": RUN_TIME,
                   "regimes": [{"name": r["name"], "k": r["k"]} for r in REGIMES]},
    })

    try:
        core = build_core()
        results = {}
        for cfg in REGIMES:
            print(
                f"[transient-diffusion] running {cfg['name']} ({cfg['label']}): "
                f"D={D} k={cfg['k']} resolution={RESOLUTION} run_time={RUN_TIME}",
                flush=True,
            )
            results[cfg["name"]] = _run_regime(core, cfg)
            r = results[cfg["name"]]
            print(
                f"[transient-diffusion] {cfg['name']} done: wall={r['wall']:.1f}s "
                f"lambda_analytic={r['lambda_analytic']:.4f} lambda_fit={r['lambda_fit']:.4f} "
                f"rel_err={r['lambda_rel_err']:.2%} c_range=[{r['c_min']:.4f},{r['c_max']:.4f}]",
                flush=True,
            )

        baseline = results["baseline"]
        shorter = results["shorter"]

        # --- expected_behavior / behavior_tests checks ---
        exponential_gradient = all(
            r["lambda_rel_err"] < LAMBDA_REL_ERR_MAX for r in results.values()
        )
        fields_bounded_positive = all(
            np.all(r["c_final"] >= -1e-6) and np.all(r["c_final"] <= C0 * 1.05)
            and np.all(np.isfinite(r["c_final"]))
            for r in results.values()
        )

        x_high_rel_errs = {
            name: abs(r["x_high_measured"] - r["x_high_analytic"]) / r["x_high_analytic"]
            for name, r in results.items()
        }
        french_flag_thresholds_accurate = all(
            err < X_HIGH_REL_ERR_MAX for err in x_high_rel_errs.values()
        )
        boundary_shift = baseline["x_high_measured"] - shorter["x_high_measured"]
        boundaries_shift_monotonically = (
            baseline["x_high_measured"] > results["short"]["x_high_measured"] > shorter["x_high_measured"]
            and baseline["x_low_measured"] > results["short"]["x_low_measured"] > shorter["x_low_measured"]
            and boundary_shift >= BOUNDARY_SHIFT_MIN
        )
        french_flag_thresholds = french_flag_thresholds_accurate and boundaries_shift_monotonically

        # --- viz ---
        viz_dir = STUDY_DIR / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)

        animation_html = viz.field_animation_html(
            baseline["coords"], baseline["frames"], baseline["frame_times"],
            f"Morphogen gradient forming (D={D}, k={REGIMES[0]['k']}, "
            f"lambda={baseline['lambda_analytic']:.3f}) -- DiffusionProcess ⊕ LinearDegradationProcess",
        )
        (viz_dir / "morphogen_animation.html").write_text(animation_html)

        profile_series = {}
        for cfg in REGIMES:
            r = results[cfg["name"]]
            fit_curve = C0 * np.exp(-r["x_sorted"] / r["lambda_analytic"])
            profile_series[f"{cfg['label']} (k={cfg['k']}, lambda={r['lambda_analytic']:.3f})"] = (
                r["x_sorted"], r["c_sorted"], fit_curve,
            )
        profile_html = viz.profile_compare_html(
            profile_series,
            "Steady-state morphogen gradients: lambda = sqrt(D/k) controls the range",
            x_label="x", y_label="c(x)", log_y=True,
        )
        (viz_dir / "morphogen_profile_fit.html").write_text(profile_html)

        for cfg in REGIMES:
            r = results[cfg["name"]]
            ff_html = viz.french_flag_regions_html(
                r["coords"], r["c_final"], [THETA_HIGH, THETA_LOW], REGION_LABELS,
                f"French flag readout: {cfg['label']} (k={cfg['k']}, lambda={r['lambda_analytic']:.3f})",
                boundary_x=[r["x_high_analytic"], r["x_low_analytic"]],
            )
            (viz_dir / f"morphogen_french_flag_{cfg['name']}.html").write_text(ff_html)

        # remove stale pre-rebuild diffusion-animation artifact, if present.
        stale = viz_dir / "diffusion_animation.html"
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
    print(f"[transient-diffusion] total wall-time: {total_wall:.1f}s ({total_wall / 60:.1f} min) "
          f"across {len(REGIMES)} regimes")
    for cfg in REGIMES:
        r = results[cfg["name"]]
        print(
            f"[transient-diffusion] {cfg['name']} ({cfg['label']}): k={cfg['k']} "
            f"lambda_analytic={r['lambda_analytic']:.4f} lambda_fit={r['lambda_fit']:.4f} "
            f"rel_err={r['lambda_rel_err']:.2%} wall={r['wall']:.1f}s "
            f"x_high[analytic={r['x_high_analytic']:.4f} measured={r['x_high_measured']:.4f}] "
            f"x_low[analytic={r['x_low_analytic']:.4f} measured={r['x_low_measured']:.4f}]"
        )
    print(
        f"[transient-diffusion] exponential-gradient: max rel_err="
        f"{max(r['lambda_rel_err'] for r in results.values()):.2%} "
        f"-> {'PASS' if exponential_gradient else 'FAIL'} (< {LAMBDA_REL_ERR_MAX:.0%})"
    )
    print(
        f"[transient-diffusion] fields-bounded-positive -> "
        f"{'PASS' if fields_bounded_positive else 'FAIL'}"
    )
    print(
        f"[transient-diffusion] french-flag-thresholds: x_high rel_errs={dict((n, round(e, 3)) for n, e in x_high_rel_errs.items())} "
        f"boundary_shift(baseline-shorter)={boundary_shift:.4f} monotonic={boundaries_shift_monotonically} "
        f"-> {'PASS' if french_flag_thresholds else 'FAIL'}"
    )
    print(f"[transient-diffusion] viz written: {viz_dir}")
    print(f"[transient-diffusion] run recorded in {WORKSPACE_ROOT / '.pbg' / 'runs.jsonl'}")

    all_pass = exponential_gradient and fields_bounded_positive and french_flag_thresholds
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
