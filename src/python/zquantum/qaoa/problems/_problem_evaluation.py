from functools import partial
from typing import Callable, List, Tuple, cast

import networkx as nx
import numpy as np
from zquantum.core.measurement import Measurements, expectation_values_to_real
from zquantum.core.openfermion import IsingOperator, change_operator_type
from zquantum.core.openfermion.ops.operators.qubit_operator import QubitOperator
from zquantum.core.openfermion.utils import count_qubits
from zquantum.core.utils import dec2bin


# This is the only function in this file in the Public API
def solve_problem_by_exhaustive_search(
    hamiltonian: QubitOperator,
) -> Tuple[float, List[Tuple[int, ...]]]:
    """
    Solves any QAOA cost hamiltonian using exhaustive search.

    It searches for degeneracy and returns all the best solutions if more than one
    exists.

    Args:
        hamiltonian: cost hamiltonian

    Returns:
        float: value of the best solution
        List[Tuple[int]]: list of solutions which correspond to the best value, each
            solution is a tuple of ints.
    """
    cost_function = _evaluate_solution_for_hamiltonian

    return _solve_bitstring_problem_by_exhaustive_search(
        partial(cost_function, hamiltonian=hamiltonian),
        num_nodes=count_qubits(hamiltonian),
    )


def solve_graph_problem_by_exhaustive_search(
    graph: nx.Graph,
    cost_function: Callable[[Tuple[int], nx.Graph], float],
) -> Tuple[float, List[Tuple[int, ...]]]:
    """
    Solves given graph problem using exhaustive search.

    It searches for degeneracy and returns all the best solutions if more than one
    exists.

    Args:
        graph: graph for which we want to solve the problem
        cost_function: function which calculates the cost of solution of a given
            problem.

    Returns:
        float: value of the best solution
        List[Tuple[int]]: list of solutions which correspond to the best value, each
            solution is a tuple of ints.
    """
    num_nodes = graph.number_of_nodes()
    return _solve_bitstring_problem_by_exhaustive_search(
        partial(cost_function, graph=graph), num_nodes=num_nodes
    )


def _solve_bitstring_problem_by_exhaustive_search(
    cost_function: Callable[[Tuple[int, ...]], float],
    num_nodes: int,
) -> Tuple[float, List[Tuple[int, ...]]]:
    """
    Solves given cost function of a graph problem using exhaustive search.

    It searches for degeneracy and returns all the best solutions if more than one
    exists.

    Args:
        cost_function: function which calculates the cost of solution of a given
            problem. Callable that takes a bitstring solution and outputs cost value.
        num_nodes: number of nodes of the graph for which we want to solve the problem

    Returns:
        float: value of the best solution
        List[Tuple[int]]: list of solutions which correspond to the best value, each
            solution is a tuple of ints.
    """
    solutions_list = []

    best_value = np.inf

    for i in range(2**num_nodes):
        trial_solution: Tuple[int, ...] = tuple(dec2bin(i, num_nodes))
        current_value = cost_function(trial_solution)
        if current_value == best_value:
            solutions_list.append(trial_solution)
        if current_value < best_value:
            best_value = current_value
            solutions_list = [trial_solution]

    return best_value, solutions_list


def evaluate_solution(
    solution: Tuple[int],
    graph: nx.Graph,
    get_hamiltonian: Callable[[nx.Graph], QubitOperator],
) -> float:
    """Evaluate expectation value of hamiltonian for given solution of a graph problem.

    Args:
        solution: solution to a problem as a tuple of bits
        graph: a graph for which we want to solve the problem
        get_hamiltonian: function which translates graph into a Hamiltonian representing
            a problem.

    Returns:
        float: value of a solution.
    """
    if len(graph.nodes) != len(solution):
        raise ValueError("Length of solution must match size of the graph.")
    if any(el not in [0, 1] for el in solution):
        raise ValueError("Solution must consist of either 0s or 1s.")
    return _evaluate_solution_for_hamiltonian(solution, get_hamiltonian(graph))


def _evaluate_solution_for_hamiltonian(
    solution: Tuple[int], hamiltonian: QubitOperator
) -> float:
    """Evaluates a solution of a hamiltonian by its calculating expectation value.

    Args:
        solution: solution to a problem as a tuple of bits
        hamiltonian: a Hamiltonian representing a problem.

    Returns:
        float: value of a solution.
    """
    hamiltonian = change_operator_type(hamiltonian, IsingOperator)

    expectation_values = expectation_values_to_real(
        Measurements([solution]).get_expectation_values(
            cast(IsingOperator, hamiltonian)
        )
    )
    return sum(expectation_values.values)
