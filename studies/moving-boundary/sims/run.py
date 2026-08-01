#!/usr/bin/env python3
"""Canonical run for the ``moving-boundary`` study (the deforming-domain /
prescribed-ALE showcase).

Builds the ``moving_boundary`` composite -- a single ``MovingBoundaryProcess``
(real dolfinx diffusion re-assembled and re-solved every substep on a mesh
whose geometry is mutated in place by a prescribed top-boundary law -- see
``viva_fenics/processes/moving_boundary.py``'s module docstring for the
scheme) wired to a ``RAMEmitter``:

1. Runs the baseline (oscillate, amplitude=0.3, omega=pi, one full period)
   and checks the final ``boundary_position`` against the analytic law to
   near machine precision (mirrors
   ``tests/test_moving_boundary.py::test_boundary_position_follows_oscillate_law``).
2. Runs the amplitude=0.1 and amplitude=0.5 variants and checks the
   ``domain_measure`` swing (deviation from the base area 1.0) scales with
   amplitude (mirrors
   ``tests/test_moving_boundary.py::test_domain_measure_changes_with_amplitude``).
3. Checks the diffusion field stays finite/bounded across the baseline run
   (mirrors ``tests/test_moving_boundary.py::test_field_stays_finite_and_bounded``).
4. Runs the grow-mode variant (informational only, not gated by a declared
   behavior test) to demonstrate the second boundary-motion law.
5. Renders the baseline's field as an interactive time-slider animation.

Viz coordinate caveat: ``viz.field_animation_html`` takes ONE ``coords``
array shared by every frame (see its docstring) -- it has no per-frame
coordinate support. Since the mesh genuinely deforms tick-to-tick, this run
plots every frame at the REFERENCE (t=0, undeformed unit-square) node
coordinates: the color-encoded field VALUES are exactly what was computed
on the true deformed mesh each frame, but the plotted node POSITIONS do not
themselves visually stretch/oscillate. The actual geometric deformation is
captured numerically (not spatially, in this 2D animation) by
``boundary_position``/``domain_measure``, both plotted in the console
summary below. This is the "reference-config coords" option the task brief
calls out as acceptable when the animation helper doesn't support
per-frame coordinates.

Standalone; run from the workspace root::

    pixi run python studies/moving-boundary/sims/run.py
"""
from __future__ import annotations

import math
import sys
import time
import uuid
from pathlib import Path

from process_bigraph import Composite, gather_emitter_results

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[1]

from viva_fenics import viz
from viva_fenics.core import build_core
from viva_fenics.composites.moving_boundary import moving_boundary
from viva_fenics.processes.moving_boundary import (
    _build_mesh,
    boundary_law,
    reference_coords,
)
from vivarium_workbench.lib.run_log import append_run_event

SPEC_ID = "viva_fenics.composites.moving_boundary.moving_boundary"
STUDY_SLUG = "moving-boundary"
INVESTIGATION_SLUG = "fenics-showcase"

RESOLUTION = 32
OMEGA = 3.14159
DT = 0.01
PERIOD = 2.0 * math.pi / OMEGA  # ~2.0002s -- one full oscillation

AMPLITUDE_BASELINE = 0.3
AMPLITUDE_LOW = 0.1
AMPLITUDE_HIGH = 0.5
GROW_AMPLITUDE = 0.5
GROW_TIME = 0.5

BOUNDARY_LAW_TOL = 1e-4
SWING_RATIO_MIN = 3.0
FIELD_BOUND = 5.0

# Subsample the emitter so a ~200-tick baseline run doesn't render an
# unwieldy 200-frame animation (same convention as reaction-diffusion's
# EMITTER_SUBSAMPLE).
EMITTER_SUBSAMPLE = 5


def _run(amplitude: float, run_time: float, omega: float = OMEGA, mode: str = "oscillate", subsample: int = EMITTER_SUBSAMPLE):
    core = build_core()
    doc = moving_boundary(core, resolution=RESOLUTION, amplitude=amplitude, omega=omega, dt=DT, mode=mode)
    doc["emitter"]["config"]["subsample"] = subsample
    sim = Composite({"state": doc}, core=core)
    sim.run(run_time)
    rows = gather_emitter_results(sim)[("emitter",)]

    frames, times_list, boundary_positions, domain_measures = [], [], [], []
    for i, row in enumerate(rows):
        sol = row.get("solution")
        if sol is None or len(sol) == 0:
            continue
        frames.append(sol)
        times_list.append(i * DT * subsample)
        boundary_positions.append(row.get("boundary_position"))
        domain_measures.append(row.get("domain_measure"))
    return frames, times_list, boundary_positions, domain_measures


def main() -> int:
    run_id = uuid.uuid4().hex
    started = time.time()
    n_steps_estimate = int(round(PERIOD / DT))
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
        "params": {
            "resolution": RESOLUTION,
            "amplitude": AMPLITUDE_BASELINE,
            "omega": OMEGA,
            "dt": DT,
            "period": PERIOD,
            "amplitude_sweep": [AMPLITUDE_LOW, AMPLITUDE_HIGH],
        },
    })

    try:
        # --- 1+3: baseline over one full oscillation period ---
        frames, times_list, boundary_positions, domain_measures = _run(AMPLITUDE_BASELINE, PERIOD)
        if len(frames) < 2:
            print("[moving-boundary] ERROR: fewer than 2 solution frames recorded", file=sys.stderr)
            append_run_event(WORKSPACE_ROOT, {
                "run_id": run_id,
                "event": "completed",
                "completed_at": time.time(),
                "n_steps": len(frames),
                "status": "failed",
            })
            return 1

        t_final = times_list[-1]
        boundary_position_final = boundary_positions[-1]
        analytic_final = boundary_law("oscillate", t_final, AMPLITUDE_BASELINE, OMEGA)
        boundary_law_abs_error = abs(boundary_position_final - analytic_final)
        boundary_follows_law = boundary_law_abs_error < BOUNDARY_LAW_TOL

        max_abs_field = max(max(abs(v) for v in frame) for frame in frames)
        field_finite = max_abs_field < FIELD_BOUND

        # --- 2: domain_measure swing across the amplitude sweep ---
        _f_low, _t_low, _bp_low, dm_low = _run(AMPLITUDE_LOW, PERIOD)
        _f_high, _t_high, _bp_high, dm_high = _run(AMPLITUDE_HIGH, PERIOD)
        swing_low = max(abs(v - 1.0) for v in dm_low)
        swing_high = max(abs(v - 1.0) for v in dm_high)
        swing_ratio = swing_high / max(swing_low, 1e-12)
        swing_scales_with_amplitude = swing_ratio >= SWING_RATIO_MIN

        # --- 4: grow-mode demonstration (informational, not behavior-tested) ---
        _f_grow, t_grow, bp_grow, dm_grow = _run(
            GROW_AMPLITUDE, GROW_TIME, mode="grow", subsample=1,
        )
        grow_final_position = bp_grow[-1] if bp_grow else float("nan")
        grow_monotonic = all(b > a for a, b in zip(bp_grow, bp_grow[1:]))

        # --- 5: viz (reference-config coords; see module docstring) ---
        _, V, _y_ref = _build_mesh(RESOLUTION, degree=1)
        coords = reference_coords(V)

        viz_dir = STUDY_DIR / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        html = viz.field_animation_html(
            coords, frames, times_list,
            f"Diffusion on a deforming domain (amplitude={AMPLITUDE_BASELINE}, omega={OMEGA:.3f}, "
            f"mode=oscillate) -- prescribed-ALE",
        )
        (viz_dir / "moving_boundary.html").write_text(html)
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
        "n_steps": len(frames),
        "status": "completed",
    })

    # --- report: PASS/FAIL mirrors this study's declared behavior_tests
    # exactly (boundary-follows-prescribed-law, domain-measure-scales-with-
    # amplitude, field-stays-finite) -- no additional undeclared conditions.
    print(
        f"[moving-boundary] baseline amplitude={AMPLITUDE_BASELINE} omega={OMEGA:.5f} resolution={RESOLUTION}: "
        f"boundary_position(t={t_final:.4f})={boundary_position_final:.8f} vs analytic={analytic_final:.8f} "
        f"(abs error {boundary_law_abs_error:.2e}) "
        f"-> {'PASS' if boundary_follows_law else 'FAIL'} (boundary-follows-prescribed-law < {BOUNDARY_LAW_TOL})"
    )
    print(
        f"[moving-boundary] domain_measure swing: amplitude={AMPLITUDE_LOW}->{swing_low:.4f}, "
        f"amplitude={AMPLITUDE_HIGH}->{swing_high:.4f}, ratio={swing_ratio:.2f}x "
        f"-> {'PASS' if swing_scales_with_amplitude else 'FAIL'} "
        f"(domain-measure-scales-with-amplitude >= {SWING_RATIO_MIN}x)"
    )
    print(
        f"[moving-boundary] max|field| over baseline run: {max_abs_field:.4f} "
        f"-> {'PASS' if field_finite else 'FAIL'} (field-stays-finite < {FIELD_BOUND})"
    )
    print(
        f"[moving-boundary] mode='grow' demo (amplitude={GROW_AMPLITUDE}, t=0..{GROW_TIME}): "
        f"boundary_position -> {grow_final_position:.4f} "
        f"({'monotonically increasing' if grow_monotonic else 'NOT monotonic -- unexpected'}) [informational, not gated]"
    )
    print(f"[moving-boundary] viz written: {viz_dir / 'moving_boundary.html'}")
    print(f"[moving-boundary] run recorded in {WORKSPACE_ROOT / '.pbg' / 'runs.jsonl'}")

    all_pass = boundary_follows_law and swing_scales_with_amplitude and field_finite

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
