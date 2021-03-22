import openfermion
import networkx as nx
import numpy as np
from qiskit.optimization.applications.ising import graph_partition
from zquantum.core.openfermion import qiskitpauli_to_qubitop


def _adjacency_matrix(graph: nx.Graph) -> np.ndarray:
    return nx.to_numpy_array(graph)


def get_graph_partition_hamiltonian(graph: nx.Graph) -> openfermion.QubitOperator:
    weight_matrix = _adjacency_matrix(graph)
    # TODO: add offset to QiskitOperator
    qiskit_operator, offset = graph_partition.get_operator(weight_matrix)
    return qiskitpauli_to_qubitop(qiskit_operator) + offset
