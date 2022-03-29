import networkx as nx
from zquantum.core.openfermion import QubitOperator

from .problem import Problem


class GraphPartitioning(Problem):
    """Solves the graph partitioning problem for undirected, unweighted graph using
    an ising model formulation.

    The solution of a graph partitioning  problem is the set of nodes S that minimizes
    the weight of the edges between S and the rest of the graph. S must contain half
    the nodes in the graph.

    From "Ising formulations of many NP problems" by A. Lucas, page 6, eqs 7-9
    (https://arxiv.org/pdf/1302.5843.pdf).
    """

    def _build_hamiltonian(self, graph: nx.Graph) -> QubitOperator:
        """Construct a qubit operator with Hamiltonian for the graph partition problem.

        The returned Hamiltonian is consistent with the definitions from
        "Ising formulations of many NP problems" by A. Lucas, page 6
        (https://arxiv.org/pdf/1302.5843.pdf).

        The operator's terms contain Pauli Z matrices applied to qubits. The qubit
        indices are based on graph node indices in the graph definition, not on the
        node names.

        Args:
            graph: undirected weighted graph defining the problem
        """
        ham_a = QubitOperator()
        for i in graph.nodes:
            ham_a += QubitOperator(f"Z{i}")
        ham_a = ham_a ** 2

        ham_b = QubitOperator()
        for i, j in graph.edges:
            ham_b += 1 - QubitOperator(f"Z{i} Z{j}")
        ham_b /= 2

        return ham_a + ham_b
