from typing import Dict, List, Tuple
from openfermion import QubitOperator
import networkx as nx
import numpy as np
from qiskit.optimization.applications.ising import vertex_cover
from ._qiskit_wrapper import get_hamiltonian_for_problem
from ._problem_evaluation import (
    solve_graph_problem_by_exhaustive_search,
    evaluate_solution,
)


def get_vertex_cover_hamiltonian(
    graph: nx.Graph, scale_factor: int = 1.0, offset: int = 0.0
) -> QubitOperator:
    """Construct a qubit operator with Hamiltonian for the vertex cover problem.

    From https://arxiv.org/pdf/1302.5843.pdf, see equations 33 and 34

    The operator's terms contain Pauli Z matrices applied to qubits. The qubit indices are
    based on graph node indices in the graph definition, not on the node names.

    Args:
        graph: undirected weighted graph defining the problem
        scale_factor: constant by which all the coefficients in the Hamiltonian will be multiplied
        offset: coefficient of the constant term added to the Hamiltonian to shift its energy levels

    Returns:
        operator describing the Hamiltonian


    """
    hamiltonian = get_hamiltonian_for_problem(
        graph=graph, qiskit_operator_getter=vertex_cover.get_operator
    )
    return hamiltonian * scale_factor + offset


def evaluate_vertex_cover_solution(solution: Tuple[int], graph: nx.Graph) -> float:
    """Evaluates a solution to a vertex cover problem.

    Args:
        solution: solution to a vertex cover problem
        graph: networkx.Graph
            Input graph object.
    Returns:
        float
    """
    return evaluate_solution(solution, graph, get_vertex_cover_hamiltonian)


def solve_vertex_cover_by_exhaustive_search(
    graph: nx.Graph,
) -> Tuple[float, List[Tuple[int]]]:
    """Brute-force solver for vertex cover problem instances using exhaustive search.
    Args:
        graph (networkx.Graph): undirected weighted graph describing the problem.
        instance.

    Returns:
        tuple: tuple whose first elements is the value of solution, and second is a list of tuples
            of bit strings that correspond to the solution(s).
    """

    return solve_graph_problem_by_exhaustive_search(
        graph, cost_function=evaluate_vertex_cover_solution
    )
