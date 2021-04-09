from typing import Callable, List
import networkx as nx
import numpy as np

from zquantum.core.utils import dec2bin
from zquantum.core.measurement import Measurements, expectation_values_to_real

from openfermion import IsingOperator
from zquantum.core.openfermion import change_operator_type


def solve_graph_problem_by_exhaustive_search(
    graph: nx.Graph, cost_function: Callable, find_maximum: bool = False
):
    """
    TODO
    """
    solution_set = []
    num_nodes = len(graph.nodes)

    # find one MAXCUT solution
    sign_flipper = -1 if find_maximum else 1
    best_value = sign_flipper * np.inf

    for i in range(2 ** num_nodes):
        trial_solution = dec2bin(i, num_nodes)
        current_value = cost_function(trial_solution, graph)
        if current_value == best_value:
            solution_set.append(trial_solution)
        if current_value * sign_flipper < best_value * sign_flipper:
            best_value = current_value
            solution_set = [trial_solution]

    return best_value, solution_set


def evaluate_solution(
    solution: List[int],
    graph: nx.Graph,
    get_hamiltonian: Callable,
):
    hamiltonian = change_operator_type(get_hamiltonian(graph), IsingOperator)
    expectation_values = expectation_values_to_real(
        Measurements([solution]).get_expectation_values(hamiltonian)
    )
    return sum(expectation_values.values)
