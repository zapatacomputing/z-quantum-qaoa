from openfermion import QubitOperator
import networkx as nx

from .problem import Problem


class VertexCover(Problem):
    def get_hamiltonian(
        graph: nx.Graph,
        scale_factor: float = 1.0,
        offset: float = 0.0,
        hamiltonian_factor: int = 5,
    ) -> QubitOperator:
        """Construct a qubit operator with Hamiltonian for the vertex cover problem.

        From https://arxiv.org/pdf/1302.5843.pdf, see equations 33 and 34
        and
        https://quantumcomputing.stackexchange.com/questions/16082/vertex-cover-mappings-from-qubo-to-ising-and-vice-versa
        for corrective translation shifts

        The operator's terms contain Pauli Z matrices applied to qubits. The qubit indices are
        based on graph node indices in the graph definition, not on the node names.

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

        ham_a = QubitOperator()
        for i, j in graph.edges:
            ham_a += (1 - QubitOperator(f"Z{i}")) * (1 - QubitOperator(f"Z{j}"))
        ham_a *= hamiltonian_factor / 4

        ham_b = QubitOperator()
        for i in graph.nodes:
            ham_b += QubitOperator(f"Z{i}")
        ham_b /= 2

        hamiltonian = ham_a + ham_b + len(graph.nodes) / 2

        hamiltonian.compress()

        return hamiltonian * scale_factor + offset
