"""Composite generator coupling ``PeristalticWallProcess`` (harmonic-
extension ALE mesh motion) with ``PeristalticFlowProcess`` (ALE incompressible
Navier-Stokes) through shared bigraph stores -- peristaltic pumping BY
COMPOSITION (see ``viva_fenics.processes.peristalsis`` module docstring for
the physics and the "geometry-order coupling" mechanics).

- ``stores.mesh_displacement_y`` / ``stores.mesh_velocity_y`` are written
  (``overwrite``) by ``PeristalticWallProcess`` and read by
  ``PeristalticFlowProcess`` -- the entire mesh-motion -> flow coupling is
  this document wiring; neither process implements the other's physics.
- ``stores.phase_drive`` is a real, if usually-zero, input hook on the wall
  process (see its docstring).
- Both processes tick at the SAME ``interval`` (``dt``) so the flow process
  never uses a mesh-motion reading more than one tick stale (same
  synchronous-coupling convention as ``reaction_diffusion``'s
  DiffusionProcess ⊕ LogisticReactionProcess pairing).
"""

from __future__ import annotations

from viva_superpowers.composite_generator import composite_generator

from viva_fenics.processes.peristalsis import zero_geometry_field, zero_flow_fields


@composite_generator(
    name="peristalsis",
    description="Peristaltic pumping: traveling-wave wall occlusion ⊕ ALE Navier-Stokes.",
    parameters={
        "amplitude": {"type": "number", "default": 0.3,
                       "description": "Occlusion amplitude (max inward wall displacement)"},
        "wave_speed": {"type": "number", "default": 1.0,
                        "description": "Traveling-wave speed c"},
        "wavelength": {"type": "number", "default": 1.0,
                        "description": "Occlusion wavelength lambda"},
        "reynolds": {"type": "number", "default": 10.0,
                      "description": "Reynolds number (wave_speed * H / nu)"},
        "dt": {"type": "number", "default": 0.01,
                "description": "Shared IPCS/mesh-motion substep size (also the tick interval)"},
        "ramp_time": {"type": "number", "default": 0.5,
                       "description": "Linear 0->amplitude ramp duration (avoids an impulsive-start instability)"},
        "L": {"type": "number", "default": 2.0,
               "description": "Channel length (should be an integer multiple of wavelength)"},
        "H": {"type": "number", "default": 2.0,
               "description": "Channel height (mean, undeformed)"},
        "nx": {"type": "integer", "default": 24,
                "description": "Mesh cells along the channel length"},
        "ny": {"type": "integer", "default": 12,
                "description": "Mesh cells across the channel height"},
    },
)
def peristalsis(core=None, *, amplitude=0.3, wave_speed=1.0, wavelength=1.0,
                 reynolds=10.0, dt=0.01, ramp_time=0.5, L=2.0, H=2.0, nx=24, ny=12):
    mesh_zero = zero_geometry_field(L, H, nx, ny).tolist()
    velocity_x0, velocity_y0, pressure0 = zero_flow_fields(L, H, nx, ny)

    shared_cfg = {
        "L": L, "H": H, "nx": nx, "ny": ny,
        "amplitude": amplitude, "wavelength": wavelength, "wave_speed": wave_speed,
        "ramp_time": ramp_time, "dt": dt,
    }

    return {
        "wall": {
            "_type": "process",
            "address": "local:PeristalticWallProcess",
            "config": dict(shared_cfg),
            "inputs": {
                "phase_drive": ["stores", "phase_drive"],
            },
            "outputs": {
                "mesh_displacement_y": ["stores", "mesh_displacement_y"],
                "mesh_velocity_y": ["stores", "mesh_velocity_y"],
                "wall_min_gap": ["stores", "wall_min_gap"],
                "wall_time": ["stores", "wall_time"],
            },
            "interval": dt,
        },
        "flow": {
            "_type": "process",
            "address": "local:PeristalticFlowProcess",
            "config": dict(shared_cfg, reynolds=reynolds),
            "inputs": {
                "mesh_displacement_y": ["stores", "mesh_displacement_y"],
                "mesh_velocity_y": ["stores", "mesh_velocity_y"],
                "wall_time": ["stores", "wall_time"],
            },
            "outputs": {
                "velocity_x": ["stores", "velocity_x"],
                "velocity_y": ["stores", "velocity_y"],
                "pressure": ["stores", "pressure"],
                "mean_ux": ["stores", "mean_ux"],
                "domain_area": ["stores", "domain_area"],
                "elapsed_time": ["stores", "elapsed_time"],
            },
            "interval": dt,
        },
        "stores": {
            "phase_drive": 0.0,
            "mesh_displacement_y": mesh_zero,
            "mesh_velocity_y": mesh_zero,
            "wall_min_gap": H,
            "wall_time": 0.0,
            "velocity_x": velocity_x0.tolist(),
            "velocity_y": velocity_y0.tolist(),
            "pressure": pressure0.tolist(),
            "mean_ux": 0.0,
            "domain_area": L * H,
            "elapsed_time": 0.0,
        },
        "emitter": {
            "_type": "step",
            "address": "local:RAMEmitter",
            "config": {
                "emit": {
                    "velocity_x": "array[float]",
                    "velocity_y": "array[float]",
                    "mean_ux": "float",
                    "wall_min_gap": "float",
                    "domain_area": "float",
                    "elapsed_time": "float",
                },
            },
            "inputs": {
                "velocity_x": ["stores", "velocity_x"],
                "velocity_y": ["stores", "velocity_y"],
                "mean_ux": ["stores", "mean_ux"],
                "wall_min_gap": ["stores", "wall_min_gap"],
                "domain_area": ["stores", "domain_area"],
                "elapsed_time": ["stores", "elapsed_time"],
            },
        },
    }
