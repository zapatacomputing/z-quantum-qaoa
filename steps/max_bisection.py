import numpy as np
import scipy
import copy
import json
import matplotlib.pyplot as plt
import networkx as nx

from zquantum.core.graph import (
    save_graph,
    load_graph
)
from zquantum.core.utils import create_object

from zquantum.qaoa.graph import (
    generate_random_graph,
    get_cut_size,
    get_all_cut_sizes_of_a_graph,
    get_top_bisections_of_a_graph,
    construct_ising_hamiltonian_for_max_cut
)

from zquantum.qaoa.max_bisection import solve_max_bisection_by_fj_alg

from zquantum.qaoa.optimize import (
    compute_cvar,
    compute_pogs,
    get_binned_distribution,
    compute_binned_cvar,
    optimize_angles
)

from zquantum.qaoa.utils import (
    get_x_str,
    z_vec_to_x_vec,
    NpEncoder
)

from zquantum.core.openfermion import (
    save_qubit_operator,
    load_qubit_operator,
)

from .optimize import optimize_ansatz_based_cost_function
from .simulate import run_circuit_and_get_distribution


def generate_random_graph_and_get_seed_solutions(
    num_variables,
    topology_specs,
    weight_specs,
    num_roundings=10,
    threshold=0.9,
    local_version=False
):
    while True:
        graph = generate_random_graph(topology_specs, weight_specs)
        nx.draw(graph, with_labels=True, font_weight='bold')
        plt.show()

        laplacian = nx.laplacian_matrix(graph).todense().real
        laplacian = np.array(laplacian)

        (
            top_bisections_and_sizes, max_size, min_size, mean_size
        ) = get_top_bisections_of_a_graph(graph, ratio=threshold)
        print("\ntop bisections and size:")
        for bisection, size in top_bisections_and_sizes:
            print(f"bisection = {bisection}, size = {size}")
        print(f"max size = {max_size}")
        print(f"min size = {min_size}")
        print(f"mean size = {mean_size}\n")

        seed_bisections, seed_sizes = solve_max_bisection_by_fj_alg(
            graph,
            num_roundings=num_roundings
        )
        best_ratio = None
        best_bisection = None
        best_size = None
        print(f"{num_roundings} bisections from FJ algorithm:")
        for bisection, size in zip(seed_bisections, seed_sizes):
            ratio = (size - mean_size) / (max_size - mean_size)
            if best_ratio is None or ratio > best_ratio:
                best_ratio = ratio
                best_bisection = bisection
                best_size = size
            print(f"bisection = {bisection}, size = {size}")
        print(f"best approx ratio = {best_ratio}")
        print(f"best bisection = {best_bisection}, size = {best_size}\n")
        if best_ratio < threshold:
            break

    all_sizes = get_all_cut_sizes_of_a_graph(graph, duplicate=True)

    costs = [-size for size in all_sizes]
    ranks = np.argsort(costs)

    costs_for_cvar = {get_x_str(i, num_variables): costs[i] for i in range(len(costs))}
    ranks_for_cvar = [get_x_str(i, num_variables) for i in ranks]

    target_operator = construct_ising_hamiltonian_for_max_cut(graph)

    costs_and_ranks = {
        "costs": costs,
        "ranks": ranks,
        "costs_for_cvar": costs_for_cvar,
        "ranks_for_cvar": ranks_for_cvar,
        "max_cost": -min_size,
        "min_cost": -max_size,
        "mean_cost": -mean_size
    }

    seed_bisections_and_sizes = {
        "seed_bisections": seed_bisections,
        "seed_sizes": seed_sizes,
        "best_bisection": best_bisection,
        "best_size": best_size,
        "best_ratio": best_ratio
    }

    if local_version:
        return graph, costs_and_ranks, seed_bisections_and_sizes, target_operator
    else:
        save_graph(graph, "graph.json")
        with open("costs-and-ranks.json", 'w') as outfile:
            json.dump(costs_and_ranks, outfile, cls=NpEncoder)
        with open("seed-bisections-and-sizes.json", 'w') as outfile:
            json.dump(seed_bisections_and_sizes, outfile, cls=NpEncoder)
        save_qubit_operator(target_operator, "target_operator.json")

def solve_max_bisection_by_cbqo(
    graph,
    num_variables,
    target_operator,
    costs_and_ranks,
    seed_bisections_and_sizes,
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
        graph = load_graph(graph)
        target_operator = load_qubit_operator(target_operator)
        with open(costs_and_ranks, 'r') as infile:
            costs_and_ranks = json.load(infile)
        with open(seed_bisections_and_sizes, 'r') as infile:
            seed_bisections_and_sizes = json.load(infile)

    laplacian = nx.laplacian_matrix(graph).todense().real
    laplacian = np.array(laplacian)

    costs = costs_and_ranks["costs"]
    costs_for_cvar = costs_and_ranks["costs_for_cvar"]
    ranks_for_cvar = costs_and_ranks["ranks_for_cvar"]
    max_cost = costs_and_ranks["max_cost"]
    min_cost = costs_and_ranks["min_cost"]
    delta = (max_cost - min_cost) / num_bins
    grid_costs = np.linspace(min_cost, max_cost-delta, num_bins)

    seed_bisections = seed_bisections_and_sizes["seed_bisections"]
    seed_sizes = seed_bisections_and_sizes["seed_sizes"]

    best_size = seed_bisections_and_sizes["best_size"]

    seed_bisection = np.array(seed_bisections[index], dtype=np.int64)
    seed_solution = z_vec_to_x_vec(seed_bisection)
    seed_size = seed_sizes[index]

    use_complete_graph = ansatz_specs["use_complete_graph"]

    value_differences = np.zeros((num_variables, num_variables))
    for i in range(num_variables):
        for j in range(num_variables):
            if i == j:
                value_difference = None
            elif seed_solution[i] == seed_solution[j]:
                value_difference = 0.0 if use_complete_graph else None
            else:
                new_bisection = copy.deepcopy(seed_bisection)
                new_bisection[i], new_bisection[j] = new_bisection[j], new_bisection[i]
                new_bisection = np.array(new_bisection)
                new_bisection_size = get_cut_size(laplacian, new_bisection)
                value_difference = new_bisection_size - seed_size
            value_differences[i][j] = value_difference

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

    opt_results, opt_params, opt_size = optimize_ansatz_based_cost_function(
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
    pogs0 = compute_pogs(dist0, -best_size-1e-6, costs_for_cvar, ranks_for_cvar)

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
    pogs = compute_pogs(dist, -best_size-1e-6, costs_for_cvar, ranks_for_cvar)

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
