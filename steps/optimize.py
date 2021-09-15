import numpy as np
import scipy
import copy

from zquantum.core.cost_function import (
    AnsatzBasedCostFunction,
)

def optimize_ansatz_based_cost_function(
    optimizer_specs,
    target_operator,
    ansatz_specs,
    backend_specs,
    estimation_method_specs,
    initial_parameters = None,
    num_trials = 10
):
    best_optimization_results = None

    for _ in range(num_trials):

        optimizer = create_object(optimizer_specs)
        ansatz = create_object(ansatz_specs)
        backend = create_object(backend_specs)
        estimation_method = create_object(estimation_method_specs)

        cost_function = AnsatzBasedCostFunction(
            target_operator,
            ansatz,
            backend,
            estimation_method=estimation_method,
            estimation_preprocessors=[],
            fixed_parameters=None,
            parameter_precision=None,
            parameter_precision_seed=None,
        )

        optimization_results = optimizer.minimize(cost_function, initial_parameters, keep_history=True)

        if best_optimization_results is None or best_optimization_results.opt_value > optimization_results.opt_value:
            best_optimization_results = optimization_results

    return (
        best_optimization_results,
        best_optimization_results.opt_params,
        best_optimization_results.opt_value
    )
