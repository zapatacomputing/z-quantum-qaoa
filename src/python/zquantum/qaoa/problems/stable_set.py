import openfermion
import networkx as nx
import numpy as np
from qiskit.optimization.applications.ising import stable_set
from ._qiskit_wrapper import get_hamiltonian_for_problem


def get_stable_set_hamiltonian(graph: nx.Graph) -> openfermion.QubitOperator:
    """Construct a qubit operator with Hamiltonian for the stable set problem.

    H = H_A + H_B
    H_A = sum\_{(i,j)\in E}{((1+ZiZj)/2)}
    H_B = sum_{i}{((-1/2) + (((1-degree(i))/2)(Zi))}

    The operator's terms contain Pauli Z matrices applied to qubits. The qubit indices are
    based on graph node indices in the graph definition, not on the node names.
    """
    return get_hamiltonian_for_problem(
        graph=graph, qiskit_operator_getter=stable_set.get_operator
    )
