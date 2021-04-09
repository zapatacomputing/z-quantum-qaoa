from typing import Dict, List, Tuple

import numpy as np
import networkx as nx
from openfermion import QubitOperator
from zquantum.core.utils import dec2bin
from ._problem_evaluation import (
    solve_graph_problem_by_exhaustive_search,
    evaluate_solution,
)
from qiskit.optimization.applications.ising import max_cut
from ._qiskit_wrapper import get_hamiltonian_for_problem
from ._generators import get_random_hamiltonians_for_problem


def get_random_maxcut_hamiltonians(
    graph_specs: Dict,
    number_of_instances: int,
    possible_number_of_qubits: List[int],
) -> List[QubitOperator]:
    """Generates random maxcut hamiltonians based on the input graph description for a range
    of number of qubits and a set number of instances.

    Args:
        graph_specs (dict): Specifications of the graph to generate. It should contain at
            least an entry with key 'type_graph' (Note: 'num_nodes' key will be overwritten)
        number_of_instances (int): The number of hamiltonians to generate
        possible_number_of_qubits (List[int]): A list containing the number of
            qubits in the hamiltonian. If it contains more than one value, then a
            random value from the list will be picked to generate each instance.

    Returns:
        List of zquantum.core.qubitoperator.QubitOperator object describing the
        Hamiltonians
        H = \\sum_{<i,j>} w_{i,j} * scaling * (Z_i Z_j - shifted * I).

    """
    return get_random_hamiltonians_for_problem(
        graph_specs,
        number_of_instances,
        possible_number_of_qubits,
        get_maxcut_hamiltonian,
    )


def get_maxcut_hamiltonian(graph: nx.Graph) -> QubitOperator:
    """Converts a MAXCUT instance, as described by a weighted graph, to an Ising
    Hamiltonian. It allows for different convention in the choice of the
    Hamiltonian.
    The returned Hamiltonian is consistent with the definitions from
    "A Quantum Approximate Optimization Algorithm" by E. Farhi, eq. 12
    (https://arxiv.org/pdf/1411.4028.pdf).

    Args:
        graph: undirected weighted graph that we needs to be solved

    Returns:
        operator describing the Hamiltonian

    """
    return get_hamiltonian_for_problem(
        graph=graph, qiskit_operator_getter=max_cut.get_operator
    )


def evaluate_maxcut_solution(solution: List[int], graph: nx.Graph) -> float:
    """Compute the Cut given a partition of the nodes.

    Args:
        solution: list[0,1]
            A list of 0-1 values indicating the partition of the nodes of a graph into two
            separate sets.
        graph: networkx.Graph
            Input graph object.
    Returns:
        float
    """

    return evaluate_solution(solution, graph, get_maxcut_hamiltonian)


def solve_maxcut_by_exhaustive_search(graph: nx.Graph) -> Tuple[float, List[List[int]]]:
    """Brute-force solver for MAXCUT instances using exhaustive search.
    Args:
        graph (networkx.Graph): undirected weighted graph describing the MAXCUT
        instance.

    Returns:
        tuple: tuple whose first elements is the number of cuts, and second is a list
            of bit strings that correspond to the solution(s).
    """

    return solve_graph_problem_by_exhaustive_search(
        graph, cost_function=evaluate_maxcut_solution, find_maximum=True
    )
