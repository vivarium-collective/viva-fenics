from process_bigraph import Composite

from viva_fenics.core import build_core


def test_build_core_resolves_local_processes():
    """build_core() must register PoissonSolverStep and DiffusionProcess in
    the core's link registry so `local:PoissonSolverStep` /
    `local:DiffusionProcess` resolve without the manual
    `core.register_link(...)` workaround tests currently need (see
    tests/test_diffusion.py's test_diffusion_composite_integral_does_not_accumulate).
    """
    core = build_core()

    assert core.link_registry["PoissonSolverStep"] is not None
    assert core.link_registry["DiffusionProcess"] is not None


def test_build_core_constructs_diffusion_composite():
    """Strongest check: build a tiny Composite document that references
    `local:DiffusionProcess` by address and confirm it constructs without a
    "cannot resolve"/"no link found" error -- the same resolution path a
    real dashboard-run Composite depends on.
    """
    core = build_core()

    document = {
        "diffusion": {
            "_type": "process",
            "address": "local:DiffusionProcess",
            "config": {"resolution": 4, "D": 0.1, "dt": 0.01},
            "inputs": {
                "source": ["stores", "source"],
                "solution": ["stores", "solution"],
            },
            "outputs": {
                "solution": ["stores", "solution"],
                "integral": ["stores", "integral"],
            },
        },
        "stores": {
            "solution": [0.0] * 25,
            "source": [0.0] * 25,
            "integral": 0.0,
        },
    }

    sim = Composite({"state": document}, core=core)
    assert sim is not None


def test_build_core_constructs_poisson_composite():
    """Same resolution check for the local:PoissonSolverStep address."""
    core = build_core()

    document = {
        "poisson": {
            "_type": "step",
            "address": "local:PoissonSolverStep",
            "config": {"resolution": 4, "degree": 1},
            "inputs": {},
            "outputs": {
                "solution": ["stores", "solution"],
                "l2_error": ["stores", "l2_error"],
            },
        },
        "stores": {
            "solution": [],
            "l2_error": 0.0,
        },
    }

    sim = Composite({"state": document}, core=core)
    assert sim is not None
