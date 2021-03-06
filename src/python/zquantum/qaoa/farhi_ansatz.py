from zquantum.core.interfaces.ansatz import Ansatz, ansatz_property
from zquantum.core.circuit import Circuit, Qubit, create_layer_of_gates
from zquantum.core.evolution import time_evolution
from zquantum.core.openfermion import qubitop_to_pyquilpauli

from .utils import create_all_x_mixer_hamiltonian
from openfermion import QubitOperator, IsingOperator
from openfermion.utils import count_qubits
from typing import Union, Optional, List
import numpy as np
import sympy
from overrides import overrides


class QAOAFarhiAnsatz(Ansatz):

    supports_parametrized_circuits = True
    cost_hamiltonian = ansatz_property("cost_hamiltonian")
    mixer_hamiltonian = ansatz_property("mixer_hamiltonian")

    def __init__(
        self,
        number_of_layers: int,
        cost_hamiltonian: Union[QubitOperator, IsingOperator],
        mixer_hamiltonian: Optional[Union[QubitOperator, IsingOperator]] = None,
    ):
        """Ansatz class representing QAOA ansatz as described in "A Quantum Approximate Optimization Algorithm" by E. Farhi and J. Goldstone (https://arxiv.org/abs/1411.4028)

        Args:
            number_of_layers: number of layers of the ansatz. Also refered to as "p" in the paper.
            cost_hamiltonian: Hamiltonian representing the cost function
            mixer_hamiltonian: Mixer hamiltonian for the QAOA. If not provided, will default to basic operator consisting of single X terms.

        Attributes:
            number_of_qubits: number of qubits required for the ansatz circuit.
            number_of_params: number of the parameters that need to be set for the ansatz circuit.
        """
        super().__init__(number_of_layers)
        self._cost_hamiltonian = cost_hamiltonian
        if mixer_hamiltonian is None:
            mixer_hamiltonian = create_all_x_mixer_hamiltonian(self.number_of_qubits)
        self._mixer_hamiltonian = mixer_hamiltonian

    @property
    def number_of_qubits(self):
        """Returns number of qubits used for the ansatz circuit."""
        return count_qubits(self._cost_hamiltonian)

    @property
    def number_of_params(self) -> int:
        """Returns number of parameters in the ansatz."""
        return 2 * self.number_of_layers

    @property
    def parametrized_circuit(self) -> Circuit:
        """Returns a parametrized circuit representing QAOA ansatz."""
        if self._parametrized_circuit is None:
            if self.supports_parametrized_circuits:
                return self._generate_circuit()
            else:
                raise (
                    NotImplementedError(
                        "{0} does not support parametrized circuits.".format(
                            type(self).__name__
                        )
                    )
                )
        else:
            return self._parametrized_circuit

    @overrides
    def _generate_circuit(self, params: Optional[np.ndarray] = None) -> Circuit:
        """Returns a parametrizable circuit represention of the ansatz.
        By convention the initial state is taken to be the |+..+> state and is
        evolved first under the cost Hamiltonian and then the mixer Hamiltonian.
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
        circuit += create_layer_of_gates(self.number_of_qubits, "H")

        # Add time evolution layers
        pyquil_cost_hamiltonian = qubitop_to_pyquilpauli(self._cost_hamiltonian)
        pyquil_mixer_hamiltonian = qubitop_to_pyquilpauli(self._mixer_hamiltonian)

        for i in range(self.number_of_layers):
            circuit += time_evolution(
                pyquil_cost_hamiltonian, sympy.Symbol(f"gamma_{i}")
            )
            circuit += time_evolution(
                pyquil_mixer_hamiltonian, sympy.Symbol(f"beta_{i}")
            )

        return circuit
