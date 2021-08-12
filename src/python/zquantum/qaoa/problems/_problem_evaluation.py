from typing import Callable, List, Tuple
import networkx as nx
import numpy as np
from openfermion.ops.operators.qubit_operator import QubitOperator

from zquantum.core.utils import dec2bin
from zquantum.core.measurement import Measurements, expectation_values_to_real

from openfermion import IsingOperator
from zquantum.core.openfermion import change_operator_type


def solve_graph_problem_by_exhaustive_search(
    graph: nx.Graph,
    cost_function: Callable[[List[int], nx.Graph], float],
) -> Tuple[float, List[Tuple[int]]]:
    """
    Solves given graph problem using exhaustive search.
    It searches for degeneracy and returns all the best solutions if more than one exists.

    Args:
        graph: graph for which we want to solve the problem
        cost_function: function which calculates the cost of solution of a given problem.

    Returns:
        float: value of the best solution
        List[Tuple[int]]: list of solutions which correspond to the best value, each solution is a tuple of ints.
    """
    solutions_list = []
    num_nodes = len(graph.nodes)

    best_value = np.inf

    for i in range(2 ** num_nodes):
        trial_solution = tuple(dec2bin(i, num_nodes))
        current_value = cost_function(trial_solution, graph)
        if current_value == best_value:
            solutions_list.append(trial_solution)
        if current_value < best_value:
            best_value = current_value
            solutions_list = [trial_solution]

    return best_value, solutions_list


def evaluate_solution(
    solution: List[int],
    graph: nx.Graph,
    get_hamiltonian: Callable[[nx.Graph], QubitOperator],
) -> float:
    """Evaluates a solution of a graph problem by calculating expectation value of its Hamiltonian.

    Args:
        solution: solution to a problem
        graph: a graph for which we want to solve the problem
        get_hamiltonian: function which translates graph into a Hamiltonian representing a problem.

    Returns:
        float: value of a solution.
    """
    if len(graph.nodes) != len(solution):
        raise ValueError("Length of solution must match size of the graph.")
    if any(el not in [0, 1] for el in solution):
        raise ValueError("Solution must consist of either 0s or 1s.")
    hamiltonian = change_operator_type(get_hamiltonian(graph), IsingOperator)
    expectation_values = expectation_values_to_real(
        Measurements([solution]).get_expectation_values(hamiltonian)
    )
    return sum(expectation_values.values)
