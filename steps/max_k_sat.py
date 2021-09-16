import numpy as np
import scipy
import copy
import json

from zquantum.core.utils import create_object

from zquantum.qaoa.sat import (
    generate_random_k_sat_instance,
    construct_ising_hamiltonian_for_max_k_sat,
    save_k_sat_instance,
    load_k_sat_instance
)

from zquantum.qaoa.max_3_sat import solve_max_3_sat_by_kz_alg

from zquantum.qaoa.optimize import (
    compute_cvar,
    compute_pogs,
    get_binned_distribution,
    compute_binned_cvar,
    optimize_angles
)

from zquantum.qaoa.utils import (
    get_x_str,
    NpEncoder
)

from zquantum.core.openfermion import (
    save_qubit_operator,
    load_qubit_operator,
)

from .optimize import optimize_ansatz_based_cost_function
from .simulate import run_circuit_and_get_distribution


def generate_random_k_sat_instance_and_get_seed_solutions(
    num_variables,
    num_clauses,
    max_clause_length,
    equal_clause_length=False,
    allow_duplicates=True,
    weight_type="uniform",
    num_roundings=10,
    threshold=0.9,
    local_version=False
):
    assert max_clause_length <= 3
    while True:
        k_sat = generate_random_k_sat_instance(num_variables,
                                               num_clauses,
                                               max_clause_length,
                                               equal_clause_length=equal_clause_length,
                                               allow_duplicates=allow_duplicates,
                                               weight_type=weight_type)
        print(k_sat)

        top_assignments_and_values, max_value, min_value, mean_value = k_sat.get_top_assignments(ratio=threshold)

        print("\ntop assignments and values:")
        for assignment, value in top_assignments_and_values:
            print(f"assignment = {assignment}, value = {value}")
        print(f"max value = {max_value}")
        print(f"min value = {min_value}")
        print(f"mean value = {mean_value}\n")

        seed_assignments, seed_values = solve_max_3_sat_by_kz_alg(k_sat, num_roundings=num_roundings)
        best_ratio = None
        best_assignment = None
        best_value = None
        print(f"{num_roundings} random roundings from KZ algorithm:")
        for assignment, value in zip(seed_assignments, seed_values):
        #     ratio = best_value / max_value
            ratio = (value - mean_value) / (max_value - mean_value)
            if best_ratio is None or ratio > best_ratio:
                best_ratio = ratio
                best_assignment = assignment
                best_value = value
            print(f"assignment = {assignment}, value = {value}")
        print(f"best approx ratio = {best_ratio}")
        print(f"best assignment = {best_assignment}, value = {best_value}")
        if best_ratio < threshold:
            break

    all_values = k_sat.evaluate_all_possible_assignments()

    costs = [-value for value in all_values]
    ranks = np.argsort(costs)

    costs_for_cvar = {get_x_str(i, num_variables): costs[i] for i in range(len(costs))}
    ranks_for_cvar = [get_x_str(i, num_variables) for i in ranks]

    target_operator = construct_ising_hamiltonian_for_max_k_sat(k_sat)

    costs_and_ranks = {
        "costs": costs,
        "ranks": ranks,
        "costs_for_cvar": costs_for_cvar,
        "ranks_for_cvar": ranks_for_cvar,
        "max_cost": -min_value,
        "min_cost": -max_value,
        "mean_cost": -mean_value
    }

    seed_assignments_and_values = {
        "seed_assignments": seed_assignments,
        "seed_values": seed_values,
        "best_assignment": best_assignment,
        "best_value": best_value,
        "best_ratio": best_ratio
    }

    if local_version:
        return k_sat, costs_and_ranks, seed_assignments_and_values, target_operator
    else:
        save_k_sat_instance(k_sat, "k-sat.json")
        with open("costs-and-ranks.json", 'w') as outfile:
            json.dump(costs_and_ranks, outfile, cls=NpEncoder)
        with open("seed-assignments-and-values.json", 'w') as outfile:
            json.dump(seed_assignments_and_values, outfile, cls=NpEncoder)
        save_qubit_operator(target_operator, "target-operator.json")

def solve_max_k_sat_by_cbqo(
    k_sat,
    num_variables,
    target_operator,
    costs_and_ranks,
    seed_assignments_and_values,
    index,
    num_bins,
    ansatz_specs,
    optimizer_specs,
    backend_specs,
    estimation_method_specs,
    num_trials_for_qw = 10,
    aa_opt_method = "L-BFGS-B",
    aa_maxiter = 100,
    num_trials_for_aa = 10,
    local_version=False
):
    if not local_version:
        k_sat = load_k_sat_instance(k_sat)
        target_operator = load_qubit_operator(target_operator)
        with open(costs_and_ranks, 'r') as infile:
            costs_and_ranks = json.load(infile)
        with open(seed_assignments_and_values, 'r') as infile:
            seed_assignments_and_values = json.load(infile)

    costs = costs_and_ranks["costs"]
    costs_for_cvar = costs_and_ranks["costs_for_cvar"]
    ranks_for_cvar = costs_and_ranks["ranks_for_cvar"]
    max_cost = costs_and_ranks["max_cost"]
    min_cost = costs_and_ranks["min_cost"]
    delta = (max_cost - min_cost) / num_bins
    grid_costs = np.linspace(min_cost, max_cost-delta, num_bins)

    seed_assignments = seed_assignments_and_values["seed_assignments"]
    seed_values = seed_assignments_and_values["seed_values"]

    best_value = seed_assignments_and_values["best_value"]

    seed_solution = seed_assignments[index]
    seed_value = seed_values[index]

    value_differences = []
    for i in range(num_variables):
        new_solution = copy.deepcopy(seed_solution)
        new_solution[i] = 1 - new_solution[i]
        new_value = k_sat.evaluate_assignment(new_solution)
        value_difference = new_value - seed_value
        value_differences.append(value_difference)

    num_layers = ansatz_specs["number_of_layers"]
    ansatz_specs["costs"] = costs
    ansatz_specs["seed_solution"] = seed_solution
    ansatz_specs["value_differences"] = value_differences

    current_ansatz_specs = copy.deepcopy(ansatz_specs)
    current_ansatz_specs["number_of_layers"] = 0

    alpha = estimation_method_specs["alpha"]
    estimation_method_specs["costs"] = costs_for_cvar
    estimation_method_specs["ranks"] = ranks_for_cvar

    random_initial_parameters = np.random.uniform(0.0, np.pi, 2)

    opt_results, opt_params, opt_value = optimize_ansatz_based_cost_function(
        copy.deepcopy(optimizer_specs),
        target_operator,
        copy.deepcopy(current_ansatz_specs),
        copy.deepcopy(backend_specs),
        copy.deepcopy(estimation_method_specs),
        initial_parameters = random_initial_parameters,
        num_trials = num_trials_for_qw
    )
    gamma, theta = opt_params

    dist0 = run_circuit_and_get_distribution(copy.deepcopy(current_ansatz_specs),
                                            opt_params,
                                            copy.deepcopy(backend_specs))
    cvar0 = compute_cvar(dist0, alpha, costs_for_cvar, ranks_for_cvar)
    pogs0 = compute_pogs(dist0, -best_value-1e-6, costs_for_cvar, ranks_for_cvar)

    binned_dist0 = get_binned_distribution(
        dist0,
        num_bins,
        costs_for_cvar,
        ranks_for_cvar,
        max_cost,
        min_cost
    )
    binned_cvar0 = compute_binned_cvar(grid_costs, binned_dist0, alpha)

    print(cvar0, binned_cvar0, pogs0)

    best_angles, estimated_cvar = optimize_angles(
        grid_costs,
        binned_dist0,
        num_layers,
        alpha,
        costs_for_cvar,
        ranks_for_cvar,
        opt_method = aa_opt_method,
        maxiter = aa_maxiter,
        num_trials = num_trials_for_aa
    )

    opt_params = [gamma, theta] + list(best_angles)

    print(opt_params)
    print(estimated_cvar)

    dist = run_circuit_and_get_distribution(copy.deepcopy(ansatz_specs),
                                            opt_params,
                                            copy.deepcopy(backend_specs))
    cvar = compute_cvar(dist, alpha, costs_for_cvar, ranks_for_cvar)
    pogs = compute_pogs(dist, -best_value-1e-6, costs_for_cvar, ranks_for_cvar)

    binned_dist = get_binned_distribution(
        dist,
        num_bins,
        costs_for_cvar,
        ranks_for_cvar,
        max_cost,
        min_cost
    )
    binned_cvar = compute_binned_cvar(grid_costs, binned_dist, alpha)

    print(cvar, binned_cvar, pogs)

    result = {
        "opt_params": opt_params,
        "initial_state": {
                "dist": dist0,
                "cvar": cvar0,
                "pogs": pogs0,
                "binned_cvar": binned_cvar0
        },
        "final_state": {
                "dist": dist,
                "cvar": cvar,
                "pogs": pogs,
                "binned_cvar": binned_cvar
        }
    }

    if local_version:
        return result
    else:
        with open("result.json", 'w') as outfile:
            json.dump(result, outfile)
