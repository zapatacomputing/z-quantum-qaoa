import networkx as nx
from openfermion import QubitOperator

from .problem import Problem


class VertexCover(Problem):
    """Solves vertex cover problem. 
    
    The solution of a vertex cover problem is the minimal number of colored
    verticies such that all edges connect to a colored vertex.
    From "Ising formulations of many NP Problems" by A. Lucas, eq. 33 and 34
    (https://arxiv.org/pdf/1302.5843.pdf)
    and
    https://quantumcomputing.stackexchange.com/questions/16082/vertex-cover-mappings-from-qubo-to-ising-and-vice-versa
    for corrective translation shifts
    
    Attributes:
        _A: Hard constraint with huge penalty ensuring output is a solution.
        _B: Cost of coloring a particular vertex.
    """
    def __init__(self, A: int = 5, B: int = 1):
        self._A = A
        self._B = B

    def _build_hamiltonian(
        self,
        graph: nx.Graph,
    ) -> QubitOperator:
        """Construct a Hamiltonian for the vertex cover problem.

        The operator's terms contain Pauli Z matrices applied to qubits. The qubit
        indices are based on graph node indices in the graph definition, not on the
        node names.

        Args:
            graph: undirected weighted graph defining the problem

        """  # noqa: E501
        ham_a = QubitOperator()
        for i, j in graph.edges:
            ham_a += (1 - QubitOperator(f"Z{i}")) * (1 - QubitOperator(f"Z{j}"))
        ham_a *= self._A / 4

        ham_b = QubitOperator()
        for i in graph.nodes:
            ham_b += QubitOperator(f"Z{i}")
        ham_b *= self._B / 2

        return ham_a + ham_b + len(graph.nodes) * self._B / 2
