import openfermion
import networkx as nx
import numpy as np
from qiskit.optimization.applications.ising import graph_partition
from zquantum.core.openfermion import qiskitpauli_to_qubitop
from ._qiskit_problem_helpers import _get_hamiltonian_for_problem


def get_graph_partition_hamiltonian(graph: nx.Graph) -> openfermion.QubitOperator:
    """Construct a qubit operator with Hamiltonian for the graph partition problem.

    The returned Hamiltonian is consistent with the definitions from
    "Ising formulations of many NP problems" by A. Lucas, page 6
    (https://arxiv.org/pdf/1302.5843.pdf).

    The operator's terms contain Pauli Z matrices applied to qubits. The qubit indices are
    based on graph node indices in the graph definition, not on the node names.
    """
    return _get_hamiltonian_for_problem(graph=graph, problem_type="graph_partition")
