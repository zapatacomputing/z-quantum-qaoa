from typing import Optional, Union

import numpy as np
import sympy
from overrides import overrides
from zquantum.core.circuits import RY, RZ, Circuit, create_layer_of_gates
from zquantum.core.evolution import time_evolution
from zquantum.core.interfaces.ansatz import Ansatz, ansatz_property
from zquantum.core.openfermion import IsingOperator, QubitOperator, change_operator_type
from zquantum.core.openfermion.utils import count_qubits


class WarmStartQAOAAnsatz(Ansatz):

    supports_parametrized_circuits = True
    cost_hamiltonian = ansatz_property("cost_hamiltonian")
    thetas = ansatz_property("thetas")

    def __init__(
        self,
        number_of_layers: int,
        cost_hamiltonian: Union[QubitOperator, IsingOperator],
        thetas: np.ndarray,
    ):
        """Implementation of the warm-start QAOA Ansatz.

        Original paper: https://arxiv.org/abs/2009.10095v3 .

        This implementation has slightly modified mixer Hamiltonian and initial state,
        which are based on the relaxed (i.e. allowing for continuous values) solution
        of the problem defined by Ising Hamiltonian.

        Args:
            number_of_layers: number of layers of the ansatz. Also refered to as "p" in
               the paper.
            cost_hamiltonian: Hamiltonian representing the cost function
            thetas: array of floats representing angles based on the solution of
               relaxed problems.

        Attributes:
            number_of_qubits: number of qubits required for the ansatz circuit.
            number_of_params: number of the parameters that need to be set for the
                ansatz circuit.
        """
        super().__init__(number_of_layers)
        self._cost_hamiltonian = cost_hamiltonian
        self._thetas = thetas

    @property
    def number_of_qubits(self):
        """Returns number of qubits used for the ansatz circuit."""
        return count_qubits(change_operator_type(self._cost_hamiltonian, QubitOperator))

    @property
    def number_of_params(self) -> int:
        """Returns number of parameters in the ansatz."""
        return 2 * self.number_of_layers

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
        circuit = Circuit()

        # Prepare initial state
        circuit += create_layer_of_gates(self.number_of_qubits, RY, self._thetas)

        # Add time evolution layers
        cost_circuit = time_evolution(
            change_operator_type(self._cost_hamiltonian, QubitOperator),
            sympy.Symbol("gamma"),
        )
        for i in range(self.number_of_layers):
            circuit += cost_circuit.bind(
                {sympy.Symbol("gamma"): sympy.Symbol(f"gamma_{i}")}
            )
            circuit += create_layer_of_gates(self.number_of_qubits, RY, -self._thetas)
            circuit += create_layer_of_gates(
                self.number_of_qubits,
                RZ,
                np.array([-2 * sympy.Symbol(f"beta_{i}")] * self.number_of_qubits),
            )
            circuit += create_layer_of_gates(self.number_of_qubits, RY, self._thetas)

        return circuit


def convert_relaxed_solution_to_angles(
    relaxed_solution: np.ndarray, epsilon: float = 0.5
) -> np.ndarray:
    """
    Maps solution to a QP problem from values between 0 to 1 to values between 0-2pi.
    It uses method presented in section 2B in https://arxiv.org/abs/2009.10095v3 .

    Args:
        relaxed_solution: relaxed solution.
        epsilon: regularization constant.

    Returns:
        np.ndarray: converted values.
    """
    if not ((relaxed_solution >= 0) & (relaxed_solution <= 1)).all():
        raise ValueError("Relaxed solution must consist of values between 0 and 1.")

    converted_solution = []
    for value in relaxed_solution:
        if value > epsilon and value < 1 - epsilon:
            regularized_value = 2 * np.arcsin(np.sqrt(value))
        elif value <= epsilon:
            regularized_value = 2 * np.arcsin(np.sqrt(epsilon))
        else:
            regularized_value = 2 * np.arcsin(np.sqrt(1 - epsilon))
        converted_solution.append(regularized_value)
    return np.array(converted_solution)
