from typing import List, Tuple
from openfermion import QubitOperator
import networkx as nx
from ._problem_evaluation import (
    solve_graph_problem_by_exhaustive_search,
    evaluate_solution,
)


def get_stable_set_hamiltonian(
    graph: nx.Graph, scale_factor: float = 1.0, offset: float = 0.0
) -> QubitOperator:
    """Construct a qubit operator with Hamiltonian for the stable set problem.

    Based on "Efficient Combinatorial Optimization Using Quantum Annealing" p. 8
    (https://arxiv.org/pdf/1801.08653.pdf)

    The operator's terms contain Pauli Z matrices applied to qubits. The qubit indices are
    based on graph node indices in the graph definition, not on the node names.
    Args:
        graph: undirected weighted graph defining the problem
        scale_factor: constant by which all the coefficients in the Hamiltonian will be multiplied
        offset: coefficient of the constant term added to the Hamiltonian to shift its energy levels
    """

    # Relabeling for monotonicity purposes
    num_nodes = range(len(graph.nodes))
    mapping = {node: new_label for node, new_label in zip(graph.nodes, num_nodes)}
    graph = nx.relabel_nodes(graph, mapping=mapping)

    ham_a = QubitOperator()
    for i, j in graph.edges:
        ham_a += (1 - QubitOperator(f"Z{i}")) * (1 - QubitOperator(f"Z{j}"))

    ham_b = QubitOperator()
    for i in graph.nodes:
        ham_b += QubitOperator(f"Z{i}")

    hamiltonian = ham_a / 2 + ham_b / 2 - len(graph.nodes) / 2

    return hamiltonian * scale_factor + offset


def evaluate_stable_set_solution(solution: Tuple[int], graph: nx.Graph) -> float:
    """Evaluates a solution to a graph partition problem.

    Args:
        solution: solution to a graph partition problem
        graph: networkx.Graph
            Input graph object.
    """
    return evaluate_solution(solution, graph, get_stable_set_hamiltonian)


def solve_stable_set_by_exhaustive_search(
    graph: nx.Graph,
) -> Tuple[float, List[Tuple[int]]]:
    """Brute-force solver for Graph Partition problem instances using exhaustive search.
    Args:
        graph (networkx.Graph): undirected weighted graph describing the problem.
        instance.

    Returns:
        tuple: tuple whose first elements is the value of solution, and second is a list of tuples
            of bit strings that correspond to the solution(s).
    """

    return solve_graph_problem_by_exhaustive_search(
        graph, cost_function=evaluate_stable_set_solution
    )
