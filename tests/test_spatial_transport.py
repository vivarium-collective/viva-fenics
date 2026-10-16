"""The two spatial_transport face adapters: port contract, dolfinx-free
import-safety, and a real dolfinx solver run.

These classes are the ``spatial_transport`` face the viva-genesis vent
(og-00) and thermophoretic-trap (og-03) studies consume. They drive
viva-fenics's real ``advection_diffusion_step`` solver but keep the dolfinx
import lazy, so a consumer workspace can import + register them in a
dolfinx-free ``uv`` venv and only pay the conda dependency when actually
running the FEM arm under pixi.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from viva_fenics.processes.spatial_transport import (
    FenicsReactiveTransportProcess,
    FenicsThermophoreticTrapProcess,
)

# The exact port names the consuming workspace's spatial_transport face binds
# to. These must not drift -- a composite swaps only the address and expects
# these ports verbatim (assert_face checks this at build time in viva-genesis).
_REACTIVE_INPUTS = {"temperature", "pH", "porosity", "H2", "alkalinity",
                    "heat", "flow_velocity", "pH_eq"}
_REACTIVE_OUTPUTS = {"temperature", "pH", "porosity", "flux_divergence_residual"}
_TRAP_INPUTS = {"temperature", "flow_velocity", "porosity"}
_TRAP_OUTPUTS = {"enrichment_factor", "residence_time", "accumulation_peclet",
                 "flux_divergence_residual"}


def test_reactive_transport_face_ports():
    from process_bigraph import allocate_core
    p = FenicsReactiveTransportProcess(config={}, core=allocate_core())
    assert set(p.inputs()) == _REACTIVE_INPUTS
    assert set(p.outputs()) == _REACTIVE_OUTPUTS
    # flux_divergence_residual is a sensor (overwrite), not an accumulating pool.
    assert p.outputs()["flux_divergence_residual"]["_type"] == "overwrite[float]"


def test_thermophoretic_trap_face_ports():
    from process_bigraph import allocate_core
    p = FenicsThermophoreticTrapProcess(config={}, core=allocate_core())
    assert set(p.inputs()) == _TRAP_INPUTS
    assert set(p.outputs()) == _TRAP_OUTPUTS
    for port in _TRAP_OUTPUTS:
        assert p.outputs()[port]["_type"] == "overwrite[float]"


def test_import_safe_and_registers_without_dolfinx():
    """`import viva_fenics` and registering the two classes must work with NO
    dolfinx installed (the consumer's uv .venv), and only RUNNING one there
    raises the clear 'run under pixi' error.

    Run in a subprocess with dolfinx blocked (sys.modules['dolfinx'] = None)
    so this holds even though the pixi test env HAS dolfinx.
    """
    script = textwrap.dedent(
        """
        import sys
        sys.modules["dolfinx"] = None  # simulate a dolfinx-free uv .venv

        import viva_fenics
        # the two face adapters are exposed even without dolfinx ...
        assert hasattr(viva_fenics, "FenicsReactiveTransportProcess")
        assert hasattr(viva_fenics, "FenicsThermophoreticTrapProcess")
        assert viva_fenics._DOLFINX_AVAILABLE is False
        # ... while the dolfinx-only names are absent (guarded import)
        assert not hasattr(viva_fenics, "DiffusionProcess")

        from process_bigraph import allocate_core
        core = allocate_core()
        p = viva_fenics.FenicsThermophoreticTrapProcess(config={}, core=core)
        # ports resolve (registration / assert_face path) with no dolfinx
        assert set(p.inputs()) == {"temperature", "flow_velocity", "porosity"}

        # running, however, raises a clear 'run under pixi' error, not AttributeError
        try:
            p.update(p.initial_state(), 1.0)
        except RuntimeError as exc:
            assert "pixi" in str(exc).lower()
        else:
            raise AssertionError("expected RuntimeError (run under pixi) without dolfinx")
        print("IMPORT_SAFE_OK")
        """
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert out.returncode == 0, f"stdout={out.stdout!r} stderr={out.stderr!r}"
    assert "IMPORT_SAFE_OK" in out.stdout


def test_thermophoretic_trap_drives_real_solver():
    """Real dolfinx run: the Soret drift concentrates the solute cold-side, so
    the enrichment READ from the solved field exceeds 1 (no mock, no exp(Pe))."""
    pytest.importorskip("dolfinx")
    from process_bigraph import allocate_core
    p = FenicsThermophoreticTrapProcess(config={"resolution": 48}, core=allocate_core())
    out = p.update({"temperature": 333.15, "flow_velocity": 1.0e-4, "porosity": 0.3}, 1.0)
    assert out["enrichment_factor"] > 1.0          # cold-side accumulation, from the field
    assert out["accumulation_peclet"] > 0.0
    assert out["residence_time"] > 0.0
    assert 0.0 <= out["flux_divergence_residual"] < 1.0   # converged steady residual


def test_reactive_transport_drives_real_solver():
    """Real dolfinx run: the advection-diffusion vent column emits the four
    outlet ports with finite values and a non-negative residual witness."""
    pytest.importorskip("dolfinx")
    import math

    from process_bigraph import allocate_core
    p = FenicsReactiveTransportProcess(config={"resolution": 20}, core=allocate_core())
    state = {"temperature": 300.0, "pH": 7.2, "porosity": 0.35,
             "H2": 1.0e-3, "alkalinity": 1.0e-3, "heat": 10.0,
             "flow_velocity": 1.0e-3, "pH_eq": 9.0}
    out = p.update(state, 1.0)
    assert set(out) == _REACTIVE_OUTPUTS
    # temperature/pH/porosity are first-order relaxation DELTAS (either sign);
    # assert they are finite, bounded nudges, not a specific direction.
    for k in ("temperature", "pH", "porosity", "flux_divergence_residual"):
        assert math.isfinite(out[k])
    assert out["flux_divergence_residual"] >= 0.0
    assert abs(out["temperature"]) < 55.0          # bounded by the inlet bump cap
    # alkaline serpentinized source (pH_eq=9 > interface 7.2) relaxes pH upward
    assert out["pH"] > 0.0
    # precipitation only clogs (porosity delta <= 0)
    assert out["porosity"] <= 0.0
