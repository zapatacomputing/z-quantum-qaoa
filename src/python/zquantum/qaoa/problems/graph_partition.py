from typing import Dict, List, Tuple
from openfermion import QubitOperator
import networkx as nx
import numpy as np
from qiskit.optimization.applications.ising import graph_partition
from ._qiskit_wrapper import get_hamiltonian_for_problem
from ._generators import get_random_hamiltonians_for_problem
from ._problem_evaluation import (
    solve_graph_problem_by_exhaustive_search,
    evaluate_solution,
)


def get_graph_partition_hamiltonian(graph: nx.Graph) -> QubitOperator:
    """Construct a qubit operator with Hamiltonian for the graph partition problem.

    The returned Hamiltonian is consistent with the definitions from
    "Ising formulations of many NP problems" by A. Lucas, page 6
    (https://arxiv.org/pdf/1302.5843.pdf).

    The operator's terms contain Pauli Z matrices applied to qubits. The qubit indices are
    based on graph node indices in the graph definition, not on the node names.
    """
    return get_hamiltonian_for_problem(
        graph=graph, qiskit_operator_getter=graph_partition.get_operator
    )


def get_random_graph_partition_hamiltonians(
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
        get_graph_partition_hamiltonian,
    )


def evaluate_graph_partition_solution(solution: List[int], graph: nx.Graph) -> float:
    """Evaluates a solution to a graph partition problem.

    Args:
        solution: solution to a graph partition problem
        graph: networkx.Graph
            Input graph object.
    Returns:
        float
    """

    return evaluate_solution(solution, graph, get_graph_partition_hamiltonian)


def solve_graph_partition_by_exhaustive_search(
    graph: nx.Graph,
) -> Tuple[float, List[List[int]]]:
    """Brute-force solver for Graph Partition problem instances using exhaustive search.
    Args:
        graph (networkx.Graph): undirected weighted graph describing the problem.
        instance.

    Returns:
        tuple: tuple whose first elements is the value of solution, and second is a list of lists
            of bit strings that correspond to the solution(s).
    """

    return solve_graph_problem_by_exhaustive_search(
        graph, cost_function=evaluate_graph_partition_solution, find_maximum=True
    )
