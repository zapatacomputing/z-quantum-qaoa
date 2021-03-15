from zquantum.core.interfaces.ansatz import Ansatz, ansatz_property
from zquantum.core.circuit import Circuit, Qubit, create_layer_of_gates
from zquantum.core.evolution import time_evolution
from zquantum.core.openfermion import qubitop_to_pyquilpauli, change_operator_type

from openfermion import QubitOperator, IsingOperator
from openfermion.utils import count_qubits
from typing import Union, Optional
import numpy as np
import sympy
from overrides import overrides


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
        """
        This is implementationg of the warm-start QAOA Ansatz from https://arxiv.org/abs/2009.10095v3 .
        It has slightly modified mixer Hamiltonian and initial state, which are based on the relaxed
        (i.e. allowing for continuous values) solution of the problem defined by Ising Hamiltonian.

        Args:
            number_of_layers: number of layers of the ansatz. Also refered to as "p" in the paper.
            cost_hamiltonian: Hamiltonian representing the cost function
            thetas: array of floats representing angles based on the solution of relaxed problems.

        Attributes:
            number_of_qubits: number of qubits required for the ansatz circuit.
            number_of_params: number of the parameters that need to be set for the ansatz circuit.
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
        qubits = [Qubit(qubit_index) for qubit_index in range(self.number_of_qubits)]
        circuit.qubits = qubits

        # Prepare initial state
        circuit += create_layer_of_gates(self.number_of_qubits, "Ry", self._thetas)

        pyquil_cost_hamiltonian = qubitop_to_pyquilpauli(
            change_operator_type(self._cost_hamiltonian, QubitOperator)
        )

        # Add time evolution layers
        for i in range(self.number_of_layers):
            circuit += time_evolution(
                pyquil_cost_hamiltonian, sympy.Symbol(f"gamma_{i}")
            )
            circuit += create_layer_of_gates(self.number_of_qubits, "Ry", -self._thetas)
            circuit += create_layer_of_gates(
                self.number_of_qubits,
                "Rz",
                [-2 * sympy.Symbol(f"beta_{i}")] * self.number_of_qubits,
            )
            circuit += create_layer_of_gates(self.number_of_qubits, "Ry", self._thetas)

        return circuit