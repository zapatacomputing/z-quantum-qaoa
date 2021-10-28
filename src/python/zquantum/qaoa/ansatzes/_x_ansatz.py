from itertools import combinations
from typing import Optional, Union

import numpy as np
import sympy
from openfermion import IsingOperator, QubitOperator
from overrides import overrides
from zquantum.core.circuits import Circuit
from zquantum.core.evolution import time_evolution
from zquantum.core.interfaces.ansatz import Ansatz, ansatz_property


class XAnsatz(Ansatz):

    supports_parametrized_circuits = True
    number_of_qubits = ansatz_property("number_of_qubits")

    def __init__(
        self,
        number_of_layers: int,
        number_of_qubits: int,
    ):
        """This is implementation of the X Ansatz from https://arxiv.org/abs/2105.01114

        Args:
            number_of_layers: k-body depth (the maximum number of qubits entangled at
                one time) as described in the original paper. Cannot be greater than
                the number of qubits.
            number_of_qubits: number of qubits required for the ansatz circuit.
        """

        super().__init__(number_of_layers)
        self.number_of_qubits = number_of_qubits

        assert number_of_layers <= number_of_qubits

    @overrides
    def _generate_circuit(self, params: Optional[np.ndarray] = None) -> Circuit:
        """Returns a parametrizable circuit represention of the ansatz.
        Args:
            params: parameters of the circuit.
        """
        if params is not None:
            Warning(
                "This method retuns a parametrizable circuit, params will be ignored."
            )

        return _create_circuit(self.number_of_layers, self.number_of_qubits, "X")


class XZAnsatz(Ansatz):

    supports_parametrized_circuits = True
    number_of_qubits = ansatz_property("number_of_qubits")
    use_k_body_z_operators = ansatz_property("use_k_body_z_operators")

    def __init__(
        self,
        number_of_layers: int,
        number_of_qubits: Union[QubitOperator, IsingOperator],
        use_k_body_z_operators: bool = True,
    ):
        """This is implementation of the XZ Ansatzes.

         Original paper: https://arxiv.org/abs/2105.01114 section 4.2

        Args:
            number_of_layers: k-body depth (the maximum number of qubits entangled at
                one time) as described in the original paper. Cannot be greater than
                the number of qubits.
            number_of_qubits: number of qubits required for the ansatz circuit.
            use_k_body_z_operators: from the two types of XZ ansatzes in the original
                paper
        """

        super().__init__(number_of_layers)
        self.number_of_qubits = number_of_qubits
        self._use_k_body_z_operators = use_k_body_z_operators

        assert number_of_layers <= number_of_qubits

    @overrides
    def _generate_circuit(self, params: Optional[np.ndarray] = None) -> Circuit:
        """Returns a parametrizable circuit represention of the ansatz.
        Args:
            params: parameters of the circuit.
        """
        if params is not None:
            Warning(
                "This method retuns a parametrizable circuit, params will be ignored."
            )

        if self._use_k_body_z_operators:
            type = "XZ1"
        else:
            type = "XZ2"
        return _create_circuit(self.number_of_layers, self.number_of_qubits, type)


def _create_circuit(k_body_depth: int, number_of_qubits: int, type: str) -> Circuit:
    """Args:
    type: X, XZ1, or XZ2
    """
    circuit = Circuit()
    j = 0

    # See equation 12 and figure 3(a) in the original paper.
    # S is a set of vertices that represents a cut of the graph while
    # A is the set of all possible cuts.
    # The maximum number of vertices in each set is the k-body depth.
    for k in range(k_body_depth):
        A = combinations(range(0, number_of_qubits), k + 1)
        for S in list(A):
            H_j = QubitOperator(" ".join([f"X{i}" for i in S]))
            circuit += time_evolution(H_j, sympy.Symbol(f"theta_{j}"))

            j += 1

            if type != "X":
                if type == "XZ1":
                    # See figure 4(a) in the original paper
                    H_k = QubitOperator(" ".join([f"Z{i}" for i in S]))
                elif type == "XZ2":
                    # See figure 4(b) in the original paper
                    H_k = QubitOperator()
                    for i in range(number_of_qubits):
                        H_k += QubitOperator((i, "Z"))

                circuit += time_evolution(H_k, sympy.Symbol(f"theta_{j}"))
                j += 1

    return circuit
