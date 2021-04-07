from typing import Callable
import openfermion
import networkx as nx
import numpy as np
from zquantum.core.openfermion import qiskitpauli_to_qubitop


def _adjacency_matrix(graph: nx.Graph) -> np.ndarray:
    return nx.to_numpy_array(graph)


def _change_matrix_to_qiskit_convention(weights_matrix: np.ndarray):
    """Terms returned by Qiskit have flipped ordering compared to what we'd expect."""
    return np.flip(weights_matrix)


def _identity_operator(coefficient: complex):
    """This is openfermion's way to encode `scalar * I` operators.

    It's only partially mentioned in the docs at
    https://quantumai.google/openfermion/tutorials/intro_to_openfermion
    """
    return openfermion.QubitOperator((), coefficient)


def get_hamiltonian_for_problem(
    graph: nx.Graph, qiskit_operator_getter: Callable
) -> openfermion.QubitOperator:
    """Construct a qubit operator with Hamiltonian for the graph partition problem.

    The returned Hamiltonian is consistent with the definitions from
    "Ising formulations of many NP problems" by A. Lucas, page 6
    (https://arxiv.org/pdf/1302.5843.pdf).

    The operator's terms contain Pauli Z matrices applied to qubits. The qubit indices are
    based on graph node indices in the graph definition, not on the node names.
    """
    weight_matrix = _adjacency_matrix(graph)
    weight_matrix_in_qiskit_convention = _change_matrix_to_qiskit_convention(
        weight_matrix
    )

    qiskit_operator, offset = qiskit_operator_getter(weight_matrix_in_qiskit_convention)

    openfermion_operator = qiskitpauli_to_qubitop(qiskit_operator)
    # openfermion's QubitOperator doesn't store the offset, we also don't have any
    # other convenient place to keep track of it, so we're adding it as a free term.
    return openfermion_operator + _identity_operator(offset)
