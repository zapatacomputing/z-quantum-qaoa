import openfermion
import networkx as nx
from qiskit.optimization.applications.ising import vertex_cover
from ._qiskit_wrapper import get_hamiltonian_for_problem


def get_vertex_cover_hamiltonian(graph: nx.Graph) -> openfermion.QubitOperator:
    """Construct a qubit operator with Hamiltonian for the vertex cover problem.

    From https://arxiv.org/pdf/1302.5843.pdf, see equations 33 and 34
    """
    return get_hamiltonian_for_problem(
        graph=graph, qiskit_operator_getter=vertex_cover.get_operator
    )
