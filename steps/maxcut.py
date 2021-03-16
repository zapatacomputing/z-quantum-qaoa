from zquantum.core.graph import generate_graph_from_specs
from zquantum.qaoa.maxcut import get_maxcut_hamiltonian, solve_maxcut_by_exhaustive_search
from zquantum.core.circuit import (
    load_circuit_template_params,
    save_circuit_template_params,
    load_parameter_grid,
    load_circuit_connectivity,
)
from zquantum.core.bitstring_distribution import save_bitstring_distribution
from zquantum.core.openfermion import load_qubit_operator
from zquantum.core.utils import create_object, load_noise_model
from zquantum.core.serialization import (
    save_optimization_results,
    load_optimization_results,
)

from zquantum.optimizers import LayerwiseAnsatzOptimizer

import networkx as nx
import yaml
import numpy as np
import os
import json
import copy

def create_and_run_qaoa_for_maxcut(graph_specs, 
    ansatz_specs, 
    backend_specs, 
    optimizer_specs, 
    cost_function_specs,
    problem_type,
    min_layer,
    max_layer,
    params_min_values,
    params_max_values,
    number_of_repeats,
    number_of_graphs):
    
    graph_output = {}
    final_results = {}
    final_parameters = {}
    final_bitstrings = {}

    for graph_id in range(number_of_graphs):
        print("Graph", graph_id)
        graph = generate_graph_from_specs(graph_specs)
        if problem_type == "maxcut":
            qubit_operator = get_maxcut_hamiltonian(
                graph, scaling=1.0, shifted=False
            )
            optimal_value, optimal_solution = solve_maxcut_by_exhaustive_search(graph)
        else:
            raise ValueError("Unknown problem type")
        graph_output[graph_id] = {}
        graph_output[graph_id]['graph'] = nx.readwrite.json_graph.node_link_data(graph)
        graph_output[graph_id]['optimal_value'] = optimal_value
        graph_output[graph_id]['optimal_solution'] = optimal_solution

        for i in range(number_of_repeats):
            print("Repeat", i)
            optimize_variational_circuit_with_layerwise_optimizer(
                copy.deepcopy(ansatz_specs),
                copy.deepcopy(backend_specs),
                copy.deepcopy(optimizer_specs),
                copy.deepcopy(cost_function_specs),
                qubit_operator,
                min_layer,
                max_layer,
                params_min_values,
                params_max_values,
            )
            os.rename("optimization-results.json", f"optimization-results-{graph_id}-{i}.json")
            os.rename("optimized-parameters.json", f"optimized-parameters-{graph_id}-{i}.json")
            os.rename("bitstring-distribution.json", f"bitstring-distribution-{graph_id}-{i}.json")

    for graph_id in range(number_of_graphs):
        opt_results_list = {}
        opt_parameters_list = {}
        bitstrings_list = {}
        for i in range(number_of_repeats):
            results_file = open(f"optimization-results-{graph_id}-{i}.json", "r")
            parameters_file = open(f"optimized-parameters-{graph_id}-{i}.json", "r")
            bitstrings_file = open(f"bitstring-distribution-{graph_id}-{i}.json", "r")
            opt_results_list[i] = yaml.load(results_file, Loader=yaml.SafeLoader)
            opt_parameters_list[i] = yaml.load(parameters_file, Loader=yaml.SafeLoader)
            bitstrings_list[i] = yaml.load(bitstrings_file, Loader=yaml.SafeLoader)
        final_results[graph_id] = opt_results_list
        final_parameters[graph_id] = opt_parameters_list
        final_bitstrings[graph_id] = bitstrings_list
        

    with open("graph-list.json", "w") as outfile:
        json.dump(graph_output, outfile)
    with open("optimization-results-aggregated.json", "w") as outfile:
        json.dump(final_results, outfile)
    with open("optimized-parameters-aggregated.json", "w") as outfile:
        json.dump(final_parameters, outfile)
    with open("bitstring-distributions-aggregated.json", "w") as outfile:
        json.dump(final_bitstrings, outfile)



def optimize_variational_circuit_with_layerwise_optimizer(
    ansatz_specs,
    backend_specs,
    optimizer_specs,
    cost_function_specs,
    qubit_operator,
    min_layer,
    max_layer,
    params_min_values,
    params_max_values,
):

    if isinstance(ansatz_specs, str):
        ansatz_specs_dict = yaml.load(ansatz_specs, Loader=yaml.SafeLoader)
    else:
        ansatz_specs_dict = ansatz_specs

    if ansatz_specs_dict["function_name"] == "QAOAFarhiAnsatz":
        ansatz = create_object(ansatz_specs_dict, cost_hamiltonian=qubit_operator)
    else:
        ansatz = create_object(ansatz_specs_dict)

    # Load optimizer specs
    if isinstance(optimizer_specs, str):
        optimizer_specs_dict = yaml.load(optimizer_specs, Loader=yaml.SafeLoader)
    else:
        optimizer_specs_dict = optimizer_specs
    optimizer = create_object(optimizer_specs_dict)

    # Load backend specs
    if isinstance(backend_specs, str):
        backend_specs_dict = yaml.load(backend_specs, Loader=yaml.SafeLoader)
    else:
        backend_specs_dict = backend_specs

    backend = create_object(backend_specs_dict)

    # Load cost function specs
    if isinstance(cost_function_specs, str):
        cost_function_specs_dict = yaml.load(
            cost_function_specs, Loader=yaml.SafeLoader
        )
    else:
        cost_function_specs_dict = cost_function_specs
    estimator_specs = cost_function_specs_dict.pop("estimator-specs", None)
    if estimator_specs is not None:
        cost_function_specs_dict["estimator"] = create_object(estimator_specs)
    cost_function_specs_dict["target_operator"] = qubit_operator
    cost_function_specs_dict["ansatz"] = ansatz
    cost_function_specs_dict["backend"] = backend
    cost_function = create_object(cost_function_specs_dict)

    lbl_optimizer = LayerwiseAnsatzOptimizer(inner_optimizer=optimizer)
    opt_results = lbl_optimizer.minimize_lbl(
        cost_function,
        min_layer=min_layer,
        max_layer=max_layer,
        params_min_values=params_min_values,
        params_max_values=params_max_values,
    )
    circuit = ansatz.get_executable_circuit(opt_results.opt_params)
    distribution = backend.run_circuit_and_measure(circuit, n_samples=10000).get_distribution()

    save_optimization_results(opt_results, "optimization-results.json")
    save_circuit_template_params(opt_results.opt_params, "optimized-parameters.json")
    save_bitstring_distribution(distribution, "bitstring-distribution.json")
