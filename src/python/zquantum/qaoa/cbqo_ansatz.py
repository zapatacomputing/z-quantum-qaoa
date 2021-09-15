from zquantum.core.interfaces.ansatz import Ansatz, ansatz_property
from zquantum.core import circuits
from openfermion import QubitOperator

from typing import List, Optional, Union

import numpy as np
import sympy
from overrides import overrides

from .utils import sigmoid

class CBQOAnsatz(Ansatz):

    supports_parametrized_circuits = True

    def __init__(
        self,
        number_of_layers: int,
        costs: np.ndarray,
        seed_solution: np.ndarray,
        value_differences: np.ndarray,
        mixer_type: str = "Grover",
    ):
        super().__init__(number_of_layers)
        assert len(costs) == 2 ** len(seed_solution)
        self._number_of_qubits = len(seed_solution)
        self.number_of_layers = number_of_layers
        self.costs = costs
        self.seed_solution = seed_solution
        assert len(value_differences) == self._number_of_qubits
        self.value_differences = value_differences
        self.mixer_type = mixer_type

    @property
    def number_of_qubits(self) -> int:
        """Returns number of qubits used for the ansatz circuit.
        """
        return self._number_of_qubits

    @property
    def number_of_params(self) -> int:
        """Returns number of parameters in the ansatz.
        """
        return 2 * self.number_of_layers + 2

    @property
    def parametrized_circuit(self) -> circuits.Circuit:
        """Returns a parametrized circuit if given ansatz supports it."""
        if self._parametrized_circuit is None:
            if self.supports_parametrized_circuits:
                self._parametrized_circuit = self._generate_circuit()
            else:
                raise (
                    NotImplementedError(
                        "{0} does not support parametrized circuits.".format(
                            type(self).__name__
                        )
                    )
                )
        return self._parametrized_circuit

    @overrides
    def _generate_circuit(self, params: Optional[np.ndarray] = None) -> circuits.Circuit:
        """Returns a parametrizable circuit represention of the ansatz.
        Args:
            params: parameters of the circuit.
        """

        n = self.number_of_qubits

        gates = []

        if params is not None:
            assert len(params) == 2 * self.number_of_layers + 2
            gamma = params[0]
            theta = params[1]
            if self.number_of_layers > 0:
                angles = params[2:]
        else:
            gamma = sympy.Symbol(f"gamma")
            theta = sympy.Symbol(f"theta")
            angles = []
            for k in range(self.number_of_layers):
                angles.append(sympy.Symbol(f"alpha_{k}"))
                angles.append(sympy.Symbol(f"beta_{k}"))

        gates += [circuits.X(i) for i in range(n) if self.seed_solution[i]==1]
        gates +=[circuits.RX(sigmoid(gamma, theta, self.value_differences[i]))(i) for i in range(n)]

        for k in range(self.number_of_layers):
            phases = [-angles[2*k]*cost for cost in self.costs]
            gates.append(circuits.MultiPhaseOperation(phases))

            gates += [circuits.RX(sigmoid(-gamma, theta, self.value_differences[i]))(i) for i in range(n)]

            gates += [circuits.X(i) for i in range(n) if self.seed_solution[i]==1]
            if self.mixer_type == "Grover":
                phases = [-angles[2*k+1]] + [0.0] * (2**n-1)
                gates.append(circuits.MultiPhaseOperation(phases))
            else:
                gates += [circuits.RZ(angles[2*k+1])(i) for i in range(n)]
            gates += [circuits.X(i) for i in range(n) if self.seed_solution[i]==1]

            gates += [circuits.RX(sigmoid(gamma, theta, self.value_differences[i]))(i) for i in range(n)]

        return circuits.Circuit(gates)
