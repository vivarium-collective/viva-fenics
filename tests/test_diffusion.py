from process_bigraph import allocate_core
from pbg_fenics.processes.diffusion import DiffusionProcess
import numpy as np


def test_diffusion_mass_and_smoothing():
    core = allocate_core()
    p = DiffusionProcess(config={"resolution": 32, "D": 0.1, "dt": 0.01}, core=core)
    s0 = np.array(p.initial_state()["solution"])
    delta = p.update({"source": np.zeros_like(s0), "solution": s0}, interval=0.05)
    s1 = s0 + np.array(delta["solution"])
    assert s1.max() < s0.max()            # peak diffuses down
    assert abs(s1.sum() - s0.sum()) / s0.sum() < 0.05   # ~mass conserved (no-flux/decay)


def test_diffusion_generator_registered():
    from viva_superpowers.composite_generator import _REGISTRY
    assert any(e.endswith(".transient_diffusion") for e in _REGISTRY)
