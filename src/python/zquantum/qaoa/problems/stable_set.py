import networkx as nx
from openfermion import QubitOperator

from .problem import Problem


class StableSet(Problem):
    def _build_hamiltonian(self, graph: nx.Graph) -> QubitOperator:
        """Construct a qubit operator with Hamiltonian for the stable set problem.

        Based on "Efficient Combinatorial Optimization Using Quantum Annealing" p. 8
        (https://arxiv.org/pdf/1801.08653.pdf)
        and also mentioned briefly in
        "Ising formulations of many NP problems" by A. Lucas, page 11 section 4.2
        (https://arxiv.org/pdf/1302.5843.pdf).

        The operator's terms contain Pauli Z matrices applied to qubits. The qubit
        indices are based on graph node indices in the graph definition, not on the
        node names.

        Args:
            graph: undirected weighted graph defining the problem
            scale_factor: constant by which all the coefficients in the Hamiltonian
                will be multiplied
            offset: coefficient of the constant term added to the Hamiltonian to shift
                its energy levels

        Returns:
            operator describing the Hamiltonian
        """
        ham_a = QubitOperator()
        for i, j in graph.edges:
            ham_a += (1 - QubitOperator(f"Z{i}")) * (1 - QubitOperator(f"Z{j}"))

        ham_b = QubitOperator()
        for i in graph.nodes:
            ham_b += QubitOperator(f"Z{i}")

        return ham_a / 2 + ham_b / 2 - len(graph.nodes) / 2
