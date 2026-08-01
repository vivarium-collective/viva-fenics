#!/usr/bin/env python3
"""Canonical run for the ``moving-boundary`` study (rebuilt as the
investigation's flagship composition showpiece: peristaltic pumping).

Builds the ``peristalsis`` composite -- ``PeristalticWallProcess`` (real
dolfinx harmonic-extension ALE mesh motion, prescribing a traveling-wave
occlusion on the channel walls) COMPOSED with ``PeristalticFlowProcess``
(real dolfinx ALE incompressible Navier-Stokes, IPCS operator splitting with
the ALE convective correction) through shared bigraph stores -- see
``viva_fenics/processes/peristalsis.py``'s module docstring for the full
physics/coupling story:

1. Runs the baseline (amplitude=0.3) and the amplitude=0.2 / amplitude=0.4
   variants, each for 4 full wave periods (period = wavelength/wave_speed =
   1.0s), and computes each run's net flow rate Q = time-average of
   ``mean_ux`` (the domain-averaged x-velocity sensor) over the POST-
   TRANSIENT window t=2.0-4.0s (discarding the startup ramp + first
   settling period).
2. Checks Q(baseline) is clearly positive (mirrors
   ``tests/test_peristalsis.py::test_composition_produces_net_positive_flow``'s
   sign check, at production scale) and Q increases monotonically across
   the amplitude sweep (mirrors
   ``tests/test_peristalsis.py::test_net_flow_grows_with_amplitude``).
3. Checks the baseline's minimum domain_area genuinely drops well below
   the undeformed L*H area (confirming the mesh really narrows).
4. Renders (a) a speed-field animation with the TRUE deforming wall
   envelope overlaid (see the viz docstring below for the moving-mesh
   caveat this inherits from the ORIGINAL prescribed-ALE showcase this
   study started from), and (b) a net-flow-rate-vs-amplitude chart.

Viz coordinate caveat (same limitation as before this study's rebuild --
see ``viva_fenics.viz.field_animation_html``'s docstring): the animation
helper takes ONE ``coords`` array shared by every frame, so the color-
encoded SPEED field is plotted at the FLOW mesh's fixed REFERENCE (t=0,
undeformed) node positions every frame -- the field VALUES are exactly
what was computed on the true deformed mesh each frame, but the plotted
node POSITIONS do not themselves visually stretch. This study's fallback
(the brief's explicitly-sanctioned option): the wall envelope curves
overlaid on top ARE plotted at their TRUE, deformed positions each frame
(a closed-form evaluation of the same occlusion law the simulation used,
not a mesh-node position), so the viewer directly sees the traveling
constriction narrowing/widening even though the background field grid
stays fixed.

Standalone; run from the workspace root::

    pixi run python studies/moving-boundary/sims/run.py
"""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

import numpy as np
from process_bigraph import Composite, gather_emitter_results

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[1]

from viva_fenics import viz
from viva_fenics.core import build_core
from viva_fenics.composites.peristalsis import peristalsis
from viva_fenics.processes.peristalsis import channel_velocity_coords, occlusion_and_rate
from vivarium_workbench.lib.run_log import append_run_event

SPEC_ID = "viva_fenics.composites.peristalsis.peristalsis"
STUDY_SLUG = "moving-boundary"
INVESTIGATION_SLUG = "fenics-showcase"

# --- shared physical/mesh parameters (see study.yaml's baseline block) ---
WAVE_SPEED = 1.0
WAVELENGTH = 1.0
PERIOD = WAVELENGTH / WAVE_SPEED  # 1.0s
REYNOLDS = 10.0
DT = 0.01
RAMP_TIME = 0.5
L, H = 2.0, 2.0
NX, NY = 24, 12

AMPLITUDE_BASELINE = 0.3
AMPLITUDE_LOW = 0.2
AMPLITUDE_HIGH = 0.4

N_PERIODS = 4
RUN_TIME = N_PERIODS * PERIOD  # 4.0s
AVERAGE_WINDOW_START = 2.0  # discard startup ramp + first settling period

Q_POSITIVE_THRESHOLD = 0.02
DOMAIN_AREA_FRACTION_THRESHOLD = 0.92

# ~400 ticks (RUN_TIME/DT) per run -- subsample so the animation stays
# within viz.field_animation_html's <=~30-frame / <=~6MB budget.
EMITTER_SUBSAMPLE = 15


def _run(amplitude: float, core):
    doc = peristalsis(
        core, amplitude=amplitude, wave_speed=WAVE_SPEED, wavelength=WAVELENGTH,
        reynolds=REYNOLDS, dt=DT, ramp_time=RAMP_TIME, L=L, H=H, nx=NX, ny=NY,
    )
    doc["emitter"]["config"]["subsample"] = EMITTER_SUBSAMPLE
    sim = Composite({"state": doc}, core=core)
    sim.run(RUN_TIME)
    rows = gather_emitter_results(sim)[("emitter",)]

    times, mean_ux, domain_area, velocity_x, velocity_y = [], [], [], [], []
    for row in rows:
        t = row.get("elapsed_time")
        if t is None:
            continue
        times.append(t)
        mean_ux.append(row.get("mean_ux"))
        domain_area.append(row.get("domain_area"))
        velocity_x.append(row.get("velocity_x"))
        velocity_y.append(row.get("velocity_y"))
    return {
        "times": np.array(times), "mean_ux": np.array(mean_ux),
        "domain_area": np.array(domain_area),
        "velocity_x": velocity_x, "velocity_y": velocity_y,
    }


def _net_flow_rate(result):
    mask = result["times"] >= AVERAGE_WINDOW_START
    if not np.any(mask):
        mask = slice(None)
    return float(np.mean(result["mean_ux"][mask]))


def main() -> int:
    run_id = uuid.uuid4().hex
    started = time.time()
    n_steps_estimate = int(round(RUN_TIME / DT)) * 3  # baseline + 2 variants
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
            "reynolds": REYNOLDS, "dt": DT, "ramp_time": RAMP_TIME,
            "L": L, "H": H, "nx": NX, "ny": NY,
            "wave_speed": WAVE_SPEED, "wavelength": WAVELENGTH,
            "amplitude_sweep": [AMPLITUDE_LOW, AMPLITUDE_BASELINE, AMPLITUDE_HIGH],
            "run_time": RUN_TIME,
        },
    })

    try:
        core = build_core()

        baseline = _run(AMPLITUDE_BASELINE, core)
        low = _run(AMPLITUDE_LOW, core)
        high = _run(AMPLITUDE_HIGH, core)

        n_frames_total = len(baseline["times"]) + len(low["times"]) + len(high["times"])
        if len(baseline["times"]) < 2:
            print("[moving-boundary] ERROR: fewer than 2 frames recorded for the baseline run", file=sys.stderr)
            append_run_event(WORKSPACE_ROOT, {
                "run_id": run_id, "event": "completed", "completed_at": time.time(),
                "n_steps": 0, "status": "failed",
            })
            return 1

        Q_low = _net_flow_rate(low)
        Q_baseline = _net_flow_rate(baseline)
        Q_high = _net_flow_rate(high)

        net_flow_positive = Q_baseline > Q_POSITIVE_THRESHOLD
        net_flow_monotonic = (Q_low < Q_baseline) and (Q_baseline < Q_high)

        # NOTE: since L spans an integer number of full wavelengths, the
        # spatial integral of the traveling wave's cos(theta) term over the
        # WHOLE channel is exactly 0 for any t -- so domain_area(t) is
        # essentially CONSTANT at L*(H-amplitude) throughout the run (a
        # real, if slightly surprising, area-conservation consequence of
        # this geometry), not something that dips to a "minimum" and
        # recovers. min() below is still a real, non-tautological reading
        # of that constant (confirmed against the undeformed L*H baseline).
        min_domain_area_baseline = float(np.min(baseline["domain_area"]))
        undeformed_area = L * H
        min_area_fraction = min_domain_area_baseline / undeformed_area
        domain_genuinely_deforms = min_area_fraction < DOMAIN_AREA_FRACTION_THRESHOLD

        # --- viz 1: speed-field animation + true deforming wall envelope ---
        coords = channel_velocity_coords(L, H, NX, NY)
        frames = [
            np.hypot(np.asarray(vx), np.asarray(vy))
            for vx, vy in zip(baseline["velocity_x"], baseline["velocity_y"])
        ]
        x_wall = np.linspace(0.0, L, 200)
        wall_top_frames, wall_bottom_frames = [], []
        for t in baseline["times"]:
            occ, _rate = occlusion_and_rate(x_wall, t, AMPLITUDE_BASELINE, WAVELENGTH, WAVE_SPEED, RAMP_TIME)
            wall_top_frames.append(H / 2.0 - occ)
            wall_bottom_frames.append(-H / 2.0 + occ)

        viz_dir = STUDY_DIR / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        anim_html = viz.field_animation_with_wall_envelope_html(
            coords, frames, baseline["times"], x_wall, wall_top_frames, wall_bottom_frames,
            f"Peristaltic pumping: |velocity| field (reference coords) + true wall envelope "
            f"(amplitude={AMPLITUDE_BASELINE}, wave_speed={WAVE_SPEED}, wavelength={WAVELENGTH})",
        )
        (viz_dir / "peristalsis_animation.html").write_text(anim_html)

        # --- viz 2: net flow rate Q vs amplitude ---
        sweep_html = viz.scalar_sweep_html(
            [AMPLITUDE_LOW, AMPLITUDE_BASELINE, AMPLITUDE_HIGH],
            [Q_low, Q_baseline, Q_high],
            "Net flow rate Q vs occlusion amplitude (peristaltic pumping)",
            x_label="occlusion amplitude", y_label="Q = time-avg(mean_ux), t=2-4s",
            annotation=f"Q increases {'monotonically' if net_flow_monotonic else 'NON-monotonically (!)'} with amplitude",
        )
        (viz_dir / "net_flow_vs_amplitude.html").write_text(sweep_html)
    except Exception:
        append_run_event(WORKSPACE_ROOT, {
            "run_id": run_id, "event": "completed", "completed_at": time.time(),
            "n_steps": 0, "status": "failed",
        })
        raise

    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id,
        "event": "completed",
        "completed_at": time.time(),
        "n_steps": n_frames_total,
        "status": "completed",
    })

    # --- report: PASS/FAIL mirrors this study's declared behavior_tests
    # exactly (net-flow-is-positive, flow-grows-with-amplitude,
    # wall-motion-genuinely-deforms-domain) -- no additional undeclared
    # conditions.
    print(
        f"[moving-boundary] peristalsis: amplitude sweep "
        f"low={AMPLITUDE_LOW} baseline={AMPLITUDE_BASELINE} high={AMPLITUDE_HIGH} "
        f"reynolds={REYNOLDS} nx={NX} ny={NY} dt={DT} "
        f"run_time={RUN_TIME}s (avg window t>={AVERAGE_WINDOW_START}s)"
    )
    print(
        f"[moving-boundary] net flow rate Q: amplitude={AMPLITUDE_LOW}->{Q_low:.5f}, "
        f"amplitude={AMPLITUDE_BASELINE}->{Q_baseline:.5f}, amplitude={AMPLITUDE_HIGH}->{Q_high:.5f} "
        f"-> {'PASS' if net_flow_monotonic else 'FAIL'} "
        f"(flow-grows-with-amplitude: Q({AMPLITUDE_LOW})<Q({AMPLITUDE_BASELINE})<Q({AMPLITUDE_HIGH}))"
    )
    print(
        f"[moving-boundary] baseline net flow rate Q={Q_baseline:.5f} "
        f"-> {'PASS' if net_flow_positive else 'FAIL'} (net-flow-is-positive > {Q_POSITIVE_THRESHOLD})"
    )
    print(
        f"[moving-boundary] baseline min domain_area={min_domain_area_baseline:.4f} "
        f"({min_area_fraction * 100:.1f}% of undeformed {undeformed_area:.2f}) "
        f"-> {'PASS' if domain_genuinely_deforms else 'FAIL'} "
        f"(wall-motion-genuinely-deforms-domain < {DOMAIN_AREA_FRACTION_THRESHOLD * 100:.0f}%)"
    )
    print(f"[moving-boundary] viz written: {viz_dir / 'peristalsis_animation.html'}")
    print(f"[moving-boundary] viz written: {viz_dir / 'net_flow_vs_amplitude.html'}")
    print(f"[moving-boundary] run recorded in {WORKSPACE_ROOT / '.pbg' / 'runs.jsonl'}")

    all_pass = net_flow_positive and net_flow_monotonic and domain_genuinely_deforms

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
