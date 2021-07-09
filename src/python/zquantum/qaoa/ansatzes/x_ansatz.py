from zquantum.core.interfaces.ansatz import Ansatz, ansatz_property

from zquantum.core.circuits import Circuit, create_layer_of_gates, H
from zquantum.core.evolution import time_evolution
from zquantum.core.openfermion import qubitop_to_pyquilpauli, change_operator_type

from openfermion import QubitOperator, IsingOperator
from openfermion.utils import count_qubits
from typing import Union, Optional, List
import numpy as np
import sympy
from overrides import overrides


class QAOAXAnsatz(Ansatz):

    supports_parametrized_circuits = True
    cost_hamiltonian = ansatz_property("cost_hamiltonian")

    def __init__(
        self,
        number_of_layers: int,
        cost_hamiltonian: Union[QubitOperator, IsingOperator],
    ):
        """This is implementation of the X Ansatz from https://arxiv.org/abs/2105.01114

        Args:
            number_of_layers: k-body-depth
            cost_hamiltonian: Hamiltonian representing the cost function

        Attributes:
            number_of_qubits: number of qubits required for the ansatz circuit.
            number_of_params: number of the parameters that need to be set for the ansatz circuit.
        """

        super().__init__(number_of_layers)
        self._cost_hamiltonian = cost_hamiltonian

    @property
    def number_of_qubits(self):
        """Returns number of qubits used for the ansatz circuit."""
        return count_qubits(change_operator_type(self._cost_hamiltonian, QubitOperator))

    @property
    def number_of_params(self) -> int:
        """Returns number of parameters in the ansatz."""
        # return sigma(self.number_of_layers self.number_of_qubits choose k
        return 2 * self.number_of_layers

    @overrides  # bc this method is in super
    def _generate_circuit(self, params: Optional(np.ndarray) = None) -> Circuit:
        """Returns a parametrizable circuit represention of the ansatz.
        Args:
            params: parameters of the circuit.
        """
        if params is not None:
            Warning(
                "This method retuns a parametrizable circuit, params will be ignored."
            )
        circuit = Circuit()

        # Add time evolution layers
        # Maybe I should make H a list of hamiltonians
        

        H = []
        S = []
        A = []
        X = np.array([[0, 1], [1, 0]])
        for j in range(self.number_of_params):
            for i in range(j):
                if sigma(1, j, ncr(self.number_of_qubits, i)) > j:
                    k = i
            if k == 1:
                S[j] = j
                # for i in range(self.number_of_qubits)
                H[j] = np.kron(X, np.eye(2), np.eye(2))
            if k == 2:
                print(ncr(self.number_of_qubits, k))

            circuit += time_evolution(H[j], sympy.Symbol(f"theta_{j}"))

        return circuit

def sigma(first, last, const):
    sum = 0
    for i in range(first, last + 1):
        sum += const * i
    return sum

# first : is the first value of (n) (the index of summation)
# last : is the last value of (n)
# const : is the number that you want to sum its multiplication each (n) times with (n)

import operator as op
from functools import reduce

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2
