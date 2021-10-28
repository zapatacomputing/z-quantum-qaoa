from openfermion import QubitOperator
import networkx as nx

from .problem import Problem


class VertexCover(Problem):
    def __init__(self, hamiltonian_factor: int = 5):
        self._hamiltonian_factor = hamiltonian_factor

    def _build_hamiltonian(
        self,
        graph: nx.Graph,
    ) -> QubitOperator:
        """Construct a qubit operator with Hamiltonian for the vertex cover problem.

        From https://arxiv.org/pdf/1302.5843.pdf, see equations 33 and 34
        and
        https://quantumcomputing.stackexchange.com/questions/16082/vertex-cover-mappings-from-qubo-to-ising-and-vice-versa
        for corrective translation shifts

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
        """  # noqa: E501
        ham_a = QubitOperator()
        for i, j in graph.edges:
            ham_a += (1 - QubitOperator(f"Z{i}")) * (1 - QubitOperator(f"Z{j}"))
        ham_a *= self._hamiltonian_factor / 4

        ham_b = QubitOperator()
        for i in graph.nodes:
            ham_b += QubitOperator(f"Z{i}")
        ham_b /= 2

        return ham_a + ham_b + len(graph.nodes) / 2
