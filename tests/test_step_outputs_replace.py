"""Regression: a Step reports results, so its outputs must replace what the store holds, not add to it.

compose-api refused all four of viva-fenics's Steps (compose-api#542, its generic check `step-outputs-replace`):
each declared bare `float`, `integer` or `array[float]` outputs, which bigraph-schema's apply ADDS into the store.
These Steps take no inputs, so in a fresh composite each fires once, but that one result still lands on top of
whatever the document seeded, and a Step fires again whenever a document wires an input to it. The fix declares
every output `overwrite[...]`, as the flow Processes already declare the values they report.
"""

import numpy as np
import pytest
from process_bigraph import Composite, gather_emitter_results

from viva_fenics.composites.poisson import poisson_baseline
from viva_fenics.core import build_core
from viva_fenics.processes.amr import AdaptiveRefinementStep
from viva_fenics.processes.flow import PorousFlowStep
from viva_fenics.processes.poisson import PoissonSolverStep
from viva_fenics.processes.pore3d import MultisolutePoreStep

STEPS = [AdaptiveRefinementStep, PorousFlowStep, PoissonSolverStep, MultisolutePoreStep]


def _seed_and_update(schema):
    """A seeded store value and an update of the port's own kind, distinct so adding and replacing differ."""
    if "array" in schema:
        return np.full(3, 10.0), np.ones(3)
    if "integer" in schema:
        return 3, 1
    return 3.0, 1.0


@pytest.mark.parametrize("step_class", STEPS, ids=lambda cls: cls.__name__)
def test_every_output_replaces_the_store(step_class):
    """Each declared output, applied by bigraph-schema onto a seeded store, leaves exactly the update."""
    core = build_core()
    step = step_class(config={}, core=core)
    for port, schema in step.outputs().items():
        seed, update = _seed_and_update(schema)
        applied, _ = core.apply(core.access(schema), seed, update)
        np.testing.assert_array_equal(applied, update, err_msg=f"{step_class.__name__}.{port}: {schema} accumulates")


def test_poisson_baseline_result_replaces_seeded_stores():
    """End to end through the shipped `poisson_baseline` composite, emitter included: with its stores seeded, the
    stores and the emitted row end with the Step's own result."""
    core = build_core()
    expected = PoissonSolverStep(config={"resolution": 4, "degree": 1}, core=core).update({})

    document = poisson_baseline(core, resolution=4, degree=1)
    document["stores"] = {"solution": np.full(len(expected["solution"]), 7.0), "l2_error": 5.0}
    sim = Composite({"state": document}, core=core)
    sim.run(1.0)

    np.testing.assert_allclose(sim.state["stores"]["solution"], expected["solution"])
    assert sim.state["stores"]["l2_error"] == pytest.approx(expected["l2_error"])
    emitted = gather_emitter_results(sim)[("emitter",)][-1]
    assert emitted["l2_error"] == pytest.approx(expected["l2_error"])
