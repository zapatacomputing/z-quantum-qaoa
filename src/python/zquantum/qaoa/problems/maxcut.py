from typing import List, Tuple

import networkx as nx
from openfermion import QubitOperator
from ._problem_evaluation import (
    solve_graph_problem_by_exhaustive_search,
    evaluate_solution,
)


def get_maxcut_hamiltonian(
    graph: nx.Graph, scale_factor: float = 1.0, offset: float = 0.0
) -> QubitOperator:
    """Converts a MAXCUT instance, as described by a weighted graph, to an Ising
    Hamiltonian. It allows for different convention in the choice of the Hamiltonian.
    The returned Hamiltonian is consistent with the definitions from
    "A Quantum Approximate Optimization Algorithm" by E. Farhi, eq. 12
    (https://arxiv.org/pdf/1411.4028.pdf)
    and
    "Performance of the Quantum Approximate Optimization Algorithm on the Maximum Cut Problem" eq. 1
    (https://arxiv.org/pdf/1811.08419.pdf)
    .

    Args:
        graph: undirected weighted graph defining the problem
        scale_factor: constant by which all the coefficients in the Hamiltonian will be multiplied
        offset: coefficient of the constant term added to the Hamiltonian to shift its energy levels

    Returns:
        operator describing the Hamiltonian

    """

    # Relabeling for monotonicity purposes
    num_nodes = range(len(graph.nodes))
    mapping = {node: new_label for node, new_label in zip(graph.nodes, num_nodes)}
    graph = nx.relabel_nodes(graph, mapping=mapping)

    hamiltonian = QubitOperator()
    shift = 0.0

    for i, j in graph.edges:
        try:
            weight = graph.adj[i][j]["weight"]
        except KeyError:
            weight = 1

        hamiltonian += weight * QubitOperator(f"Z{i} Z{j}")
        shift -= weight

    hamiltonian = 0.5 * (hamiltonian + shift)
    hamiltonian.compress()

    return hamiltonian * scale_factor + offset


def evaluate_maxcut_solution(solution: Tuple[int], graph: nx.Graph) -> float:
    """Compute the Cut given a partition of the nodes.
    In the convention we assumed, values of the cuts are negative
    to frame the problem as a minimization problem.
    So for a linear graph 0--1--2 with weights all equal 1, and the solution [0,1,0],
    the returned value will be equal to -2.

    Args:
        solution: list[0,1]
            A tuple of 0-1 values indicating the partition of the nodes of a graph into two
            separate sets.
        graph: networkx.Graph
            Input graph object.
    Returns:
        float
    """

    return evaluate_solution(solution, graph, get_maxcut_hamiltonian)


def solve_maxcut_by_exhaustive_search(
    graph: nx.Graph,
) -> Tuple[float, List[Tuple[int]]]:
    """Brute-force solver for MAXCUT instances using exhaustive search.
    Args:
        graph (networkx.Graph): undirected weighted graph describing the MAXCUT
        instance.

    Returns:
        tuple: tuple whose first elements is the number of cuts, and second is a tuple
            of bit strings that correspond to the solution(s).
    """

    return solve_graph_problem_by_exhaustive_search(
        graph, cost_function=evaluate_maxcut_solution
    )
