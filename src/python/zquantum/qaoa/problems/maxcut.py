from typing import Dict, List

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
    **kwargs
):
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


def get_maxcut_hamiltonian(
    graph: nx.Graph,
    scaling: float = 1.0,
    shifted: bool = False,
    l1_normalized: bool = False,
) -> QubitOperator:
    """Converts a MAXCUT instance, as described by a weighted graph, to an Ising
    Hamiltonian. It allows for different convention in the choice of the
    Hamiltonian.

    Args:
        graph: undirected weighted graph describing the MAXCUT
        instance.
        scaling: scaling of the terms of the Hamiltonian
        shifted: if True include a shift. Default: False
        l1_normalized: normalize the operator using the l1_norm = \\sum |w|

    Returns:
        operator describing the Hamiltonian
        H = \\sum_{<i,j>} w_{i,j} * scaling * (Z_i Z_j - shifted * I)
        or H_norm = H / l1_norm if l1_normalized is True.

    """
    return get_hamiltonian_for_problem(
        graph=graph, qiskit_operator_getter=max_cut.get_operator
    )


def evaluate_maxcut_solution(solution, graph):
    """Compute the Cut given a partition of the nodes.

    Args:
        solution: list[0,1]
            A list of 0-1 values indicating the partition of the nodes of a graph into two
            separate sets.
        graph: networkx.Graph
            Input graph object.
    """

    return evaluate_solution(solution, graph, get_maxcut_hamiltonian)


def solve_maxcut_by_exhaustive_search(graph):
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
