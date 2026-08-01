"""AdaptiveRefinementStep: a process-bigraph Step wrapping the residual-based
AMR loop in ``viva_fenics.fem_amr`` on the L-shaped re-entrant-corner
Laplace singularity.

Like ``PoissonSolverStep``, this is a Step (not a Process) -- given only its
config, it deterministically runs the WHOLE adaptive refinement sequence
(build mesh -> [solve -> estimate -> Doerfler-mark -> refine] * n_refinements
-> final solve) and reports the final solution plus the full
dofs/error convergence history, rather than modeling one time-varying tick.
"""

from process_bigraph import Step

from viva_fenics import fem_amr


class AdaptiveRefinementStep(Step):
    """Run the full residual-based Doerfler-marking AMR loop on the L-shaped
    re-entrant-corner Laplace singularity and report the convergence
    history (dofs vs energy-norm error at each level) plus the final
    solution field.
    """

    config_schema = {
        "initial_h": {"_type": "float", "_default": 0.3},
        "n_refinements": {"_type": "integer", "_default": 8},
        "marking_fraction": {"_type": "float", "_default": 0.5},
        "degree": {"_type": "integer", "_default": 1},
    }

    def inputs(self):
        return {}

    def outputs(self):
        return {
            "solution": "array[float]",
            "energy_error": "float",
            "dofs_history": "array[float]",
            "error_history": "array[float]",
        }

    def update(self, state):
        result = fem_amr.run_amr_loop(
            initial_h=self.config["initial_h"],
            n_refinements=self.config["n_refinements"],
            theta=self.config["marking_fraction"],
            degree=self.config["degree"],
        )
        return {
            "solution": result["uh"].x.array.copy(),
            "energy_error": float(result["error_history"][-1]),
            "dofs_history": result["dofs_history"],
            "error_history": result["error_history"],
        }
