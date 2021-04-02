
from copy import copy, deepcopy
from zquantum.core.interfaces.ansatz_test import AnsatzTests
from zquantum.core.circuit import Circuit, Gate, Qubit
from zquantum.core.utils import compare_unitary
from zquantum.core.openfermion import change_operator_type
from .farhi_ansatz import (
    QAOAFarhiAnsatz,
    create_farhi_qaoa_circuits,
    create_all_x_mixer_hamiltonian,
)
from openfermion import QubitOperator, IsingOperator
import pytest
import numpy as np
import sympy


# --------- contracts ---------
# Imagine this section is defined in z-quantum-core


def _validate_setting_number_of_layers(ansatz):
    new_number_of_layers = 100
    ansatz.number_of_layers = new_number_of_layers
    return ansatz.number_of_layers == new_number_of_layers


def _validate_set_number_of_layers_invalidates_parametrized_circuit(ansatz):
    new_number_of_layers = 100
    if ansatz.supports_parametrized_circuits:
        initial_circuit = ansatz.parametrized_circuit

        # When
        ansatz.number_of_layers = new_number_of_layers

        # Then
        return ansatz._parametrized_circuit is None
    else:
        return True


def _validate_number_of_params_greater_than_0(ansatz):
    if ansatz.number_of_layers != 0:
        return ansatz.number_of_params >= 0
    return True


def _validate_number_of_qubits_greater_than_0(ansatz):
    return ansatz.number_of_qubits > 0


def _validate_get_executable_circuit_is_not_empty(ansatz):
    # Given
    params = np.random.random([ansatz.number_of_params])

    # When
    circuit = ansatz.get_executable_circuit(params)

    # Then
    return len(circuit.gates) > 0

def _validate_get_executable_circuit_does_not_contain_symbols(ansatz):
    # Given
    params = np.random.random([ansatz.number_of_params])

    # When
    circuit = ansatz.get_executable_circuit(params=params)

    # Then
    return all(len(gate.symbolic_params) == 0 for gate in circuit.gates)


ANSATZ_CONTRACTS = [
    _validate_setting_number_of_layers,
    _validate_set_number_of_layers_invalidates_parametrized_circuit,
    _validate_number_of_params_greater_than_0,
    _validate_number_of_qubits_greater_than_0,
    _validate_get_executable_circuit_is_not_empty,
    _validate_get_executable_circuit_does_not_contain_symbols,
]


# --------- /contracts ---------


def _make_ansatz():
    cost_hamiltonian = QubitOperator((0, "Z")) + QubitOperator((1, "Z"))
    mixer_hamiltonian = QubitOperator((0, "X")) + QubitOperator((1, "X"))
    return QAOAFarhiAnsatz(
        number_of_layers=1,
        cost_hamiltonian=cost_hamiltonian,
        mixer_hamiltonian=mixer_hamiltonian,
    )


@pytest.mark.parametrize('contract', ANSATZ_CONTRACTS)
def test_ansatz_contract(contract):
    ansatz = _make_ansatz()
    assert contract(ansatz)


class TestFahriAnsatz:
    @pytest.fixture
    def ansatz(self):
        return _make_ansatz()

    @pytest.fixture
    def beta(self):
        return sympy.Symbol("beta_0")

    @pytest.fixture
    def gamma(self):
        return sympy.Symbol("gamma_0")

    @pytest.fixture
    def symbols_map(self, beta, gamma):
        beta_value = 0.5
        gamma_value = 0.7
        return [(beta, beta_value), (gamma, gamma_value)]

    @pytest.fixture
    def target_unitary(self, beta, gamma, symbols_map):
        target_circuit = Circuit()
        target_circuit.gates = []
        target_circuit.gates.append(Gate("H", [Qubit(0)]))
        target_circuit.gates.append(Gate("H", [Qubit(1)]))
        target_circuit.gates.append(Gate("Rz", [Qubit(0)], [2.0 * gamma]))
        target_circuit.gates.append(Gate("Rz", [Qubit(1)], [2.0 * gamma]))
        target_circuit.gates.append(Gate("Rx", [Qubit(0)], [2.0 * beta]))
        target_circuit.gates.append(Gate("Rx", [Qubit(1)], [2.0 * beta]))
        return target_circuit.evaluate(symbols_map).to_unitary()

    def test_set_cost_hamiltonian(self, ansatz):
        # Given
        new_cost_hamiltonian = QubitOperator((0, "Z")) - QubitOperator((1, "Z"))

        # When
        ansatz.cost_hamiltonian = new_cost_hamiltonian

        # Then
        assert ansatz._cost_hamiltonian == new_cost_hamiltonian

    def test_set_cost_hamiltonian_invalidates_circuit(self, ansatz):
        # Given
        new_cost_hamiltonian = QubitOperator((0, "Z")) - QubitOperator((1, "Z"))

        # When
        ansatz.cost_hamiltonian = new_cost_hamiltonian

        # Then
        assert ansatz._parametrized_circuit is None

    def test_set_mixer_hamiltonian(self, ansatz):
        # Given
        new_mixer_hamiltonian = QubitOperator((0, "Z")) - QubitOperator((1, "Z"))

        # When
        ansatz.mixer_hamiltonian = new_mixer_hamiltonian

        # Then
        ansatz._mixer_hamiltonian == new_mixer_hamiltonian

    def test_set_mixer_hamiltonian_invalidates_circuit(self, ansatz):
        # Given
        new_mixer_hamiltonian = QubitOperator((0, "Z")) - QubitOperator((1, "Z"))

        # When
        ansatz.mixer_hamiltonian = new_mixer_hamiltonian

        # Then
        assert ansatz._parametrized_circuit is None

    def test_get_number_of_qubits(self, ansatz):
        # Given
        new_cost_hamiltonian = (
            QubitOperator((0, "Z")) + QubitOperator((1, "Z")) + QubitOperator((2, "Z"))
        )
        target_number_of_qubits = 3

        # When
        ansatz.cost_hamiltonian = new_cost_hamiltonian

        # Then
        assert ansatz.number_of_qubits == target_number_of_qubits

    def test_get_number_of_qubits_with_ising_hamiltonian(self, ansatz):
        # Given
        new_cost_hamiltonian = (
            QubitOperator((0, "Z")) + QubitOperator((1, "Z")) + QubitOperator((2, "Z"))
        )
        new_cost_hamiltonian = change_operator_type(new_cost_hamiltonian, IsingOperator)
        target_number_of_qubits = 3

        # When
        ansatz.cost_hamiltonian = new_cost_hamiltonian

        # Then
        assert ansatz.number_of_qubits == target_number_of_qubits

    def test_get_parametrizable_circuit(self, ansatz, beta, gamma):
        # Then
        assert ansatz.parametrized_circuit.symbolic_params == [
            gamma,
            beta,
        ]

    def test_generate_circuit(self, ansatz, symbols_map, target_unitary):
        # When
        parametrized_circuit = ansatz._generate_circuit()
        evaluated_circuit = parametrized_circuit.evaluate(symbols_map)
        final_unitary = evaluated_circuit.to_unitary()

        # Then
        assert compare_unitary(final_unitary, target_unitary, tol=1e-10)

    def test_generate_circuit_with_ising_operator(
        self, ansatz, symbols_map, target_unitary
    ):
        # When
        ansatz.cost_hamiltonian = change_operator_type(
            ansatz.cost_hamiltonian, IsingOperator
        )

        parametrized_circuit = ansatz._generate_circuit()
        evaluated_circuit = parametrized_circuit.evaluate(symbols_map)
        final_unitary = evaluated_circuit.to_unitary()

        # Then
        assert compare_unitary(final_unitary, target_unitary, tol=1e-10)

