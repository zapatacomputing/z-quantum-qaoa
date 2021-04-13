from typing import Callable, Tuple
import openfermion
import networkx as nx
import numpy as np
from qiskit.aqua.operators import WeightedPauliOperator
from zquantum.core.openfermion import qiskitpauli_to_qubitop


def _adjacency_matrix(graph: nx.Graph) -> np.ndarray:
    return nx.to_numpy_array(graph)


def _change_matrix_to_qiskit_convention(weight_matrix: np.ndarray) -> np.ndarray:
    """Terms returned by Qiskit have flipped ordering compared to what we'd expect."""
    return np.flip(weight_matrix)


def _identity_operator(coefficient: complex) -> openfermion.QubitOperator:
    """This is openfermion's way to encode `scalar * I` operators.

    It's only partially mentioned in the docs at
    https://quantumai.google/openfermion/tutorials/intro_to_openfermion
    """
    return openfermion.QubitOperator((), coefficient)


def get_hamiltonian_for_problem(
    graph: nx.Graph,
    qiskit_operator_getter: Callable[
        [np.ndarray],
        Tuple[WeightedPauliOperator, float],
    ],
) -> openfermion.QubitOperator:
    """Construct a qubit operator with Hamiltonian for the graph partition problem, based on the qiskit implementation.
    Args:
        graph: graph for which we want to solve the problem
        qiskit_operator_getter: method which returns Hamiltonian given the weight matrix of a graph.
    Returns:
        openfermion.QubitOperator
    The operator's terms contain Pauli Z matrices applied to qubits. The qubit indices are
    based on graph node indices in the graph definition, not on the node names.
    """
    weight_matrix = _adjacency_matrix(graph)
    weight_matrix_in_qiskit_convention = _change_matrix_to_qiskit_convention(
        weight_matrix
    )

    qiskit_operator, offset = qiskit_operator_getter(weight_matrix_in_qiskit_convention)
    openfermion_operator = qiskitpauli_to_qubitop(qiskit_operator)
    # This removes all the coefficients close to 0, or its real or imag parts.
    openfermion_operator.compress()

    # openfermion's QubitOperator doesn't store the offset, we also don't have any
    # other convenient place to keep track of it, so we're adding it as a free term.
    return openfermion_operator + _identity_operator(offset)
