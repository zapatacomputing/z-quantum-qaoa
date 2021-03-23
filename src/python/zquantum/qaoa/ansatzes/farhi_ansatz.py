from zquantum.core.interfaces.ansatz import Ansatz, ansatz_property
from zquantum.core.circuit import Circuit, Qubit, create_layer_of_gates
from zquantum.core.evolution import time_evolution
from zquantum.core.openfermion import qubitop_to_pyquilpauli, change_operator_type

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
        mixer_hamiltonian: Optional[QubitOperator] = None,
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
        return count_qubits(change_operator_type(self._cost_hamiltonian, QubitOperator))

    @property
    def number_of_params(self) -> int:
        """Returns number of parameters in the ansatz."""
        return 2 * self.number_of_layers

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
        pyquil_cost_hamiltonian = qubitop_to_pyquilpauli(
            change_operator_type(self._cost_hamiltonian, QubitOperator)
        )
        pyquil_mixer_hamiltonian = qubitop_to_pyquilpauli(self._mixer_hamiltonian)

        for i in range(self.number_of_layers):
            circuit += time_evolution(
                pyquil_cost_hamiltonian, sympy.Symbol(f"gamma_{i}")
            )
            circuit += time_evolution(
                pyquil_mixer_hamiltonian, sympy.Symbol(f"beta_{i}")
            )

        return circuit


def create_farhi_qaoa_circuits(
    hamiltonians: List[QubitOperator], number_of_layers: Union[int, List[int]]
):
    """Creates parameterizable quantum circuits based on the farhi qaoa ansatz for each
    hamiltonian in the input list using the set number of layers.

    Args:
        hamiltonians (List[QubitOperator]): List of hamiltonians for constructing the
            circuits
        number_of_layers (Union[int, List[int]]): The number of layers of the ansatz in the circuit.
            If an int is passed in, the same number of layers is used for every ansatz circuit, however,
            if a list of ints is passed in, the number of layers used for the hamiltonian at index i of the hamiltonians
            list is the integer at index i of the number_of_layers list.

    Returns:
        List of zquantum.core.circuit.Circuit
    """
    if isinstance(number_of_layers, int):
        number_of_layers = [number_of_layers for _ in range(len(hamiltonians))]
    number_of_layers_list = number_of_layers
    assert len(number_of_layers_list) == len(hamiltonians)

    circuitset = []
    for number_of_layers, hamiltonian in zip(number_of_layers_list, hamiltonians):
        ansatz = QAOAFarhiAnsatz(number_of_layers, hamiltonian)
        circuitset.append(ansatz.parametrized_circuit)
    return circuitset


def create_all_x_mixer_hamiltonian(number_of_qubits):
    mixer_hamiltonian = QubitOperator()
    for i in range(number_of_qubits):
        mixer_hamiltonian += QubitOperator((i, "X"))
    return mixer_hamiltonian
