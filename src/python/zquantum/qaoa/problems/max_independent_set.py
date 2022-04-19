################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import networkx as nx
from zquantum.core.openfermion import QubitOperator

from .problem import Problem


class MaxIndependentSet(Problem):
    """Solves maximum independent set problem on an undirected graph using an
    ising model formulation.

    The solution to a maximum independent set is the largest set of nodes
    which do not share an edge.
    Based on "Efficient Combinatorial Optimization Using Quantum Annealing" p. 8
    (https://arxiv.org/pdf/1801.08653.pdf)
    and also mentioned briefly in
    "Ising formulations of many NP problems" by A. Lucas, page 11 section 4.2
    (https://arxiv.org/pdf/1302.5843.pdf).
    """

    def _build_hamiltonian(self, graph: nx.Graph) -> QubitOperator:
        """Construct a qubit operator with Hamiltonian for the maximum independent
        set problem.

        The operator's terms contain Pauli Z matrices applied to qubits. The qubit
        indices are based on graph node indices in the graph definition, not on the
        node names.

        Args:
            graph: undirected weighted graph defining the problem
        """
        ham_a = QubitOperator()
        for i, j in graph.edges:
            ham_a += (1 - QubitOperator(f"Z{i}")) * (1 - QubitOperator(f"Z{j}"))

        ham_b = QubitOperator()
        for i in graph.nodes:
            ham_b += QubitOperator(f"Z{i}")

        return ham_a / 2 + ham_b / 2 - len(graph.nodes) / 2
