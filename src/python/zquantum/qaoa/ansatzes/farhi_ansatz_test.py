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


class TestQAOAFarhiAnsatz(AnsatzTests):
    @pytest.fixture
    def ansatz(self):
        cost_hamiltonian = QubitOperator((0, "Z")) + QubitOperator((1, "Z"))
        mixer_hamiltonian = QubitOperator((0, "X")) + QubitOperator((1, "X"))
        return QAOAFarhiAnsatz(
            number_of_layers=1,
            cost_hamiltonian=cost_hamiltonian,
            mixer_hamiltonian=mixer_hamiltonian,
        )

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


def test_create_farhi_qaoa_circuits():
    # Given
    hamiltonians = [
        QubitOperator("Z0 Z1"),
        QubitOperator("Z0") + QubitOperator("Z1"),
    ]
    number_of_layers = 2

    # When
    circuits = create_farhi_qaoa_circuits(hamiltonians, number_of_layers)

    # Then
    assert len(circuits) == len(hamiltonians)

    for circuit in circuits:
        assert isinstance(circuit, Circuit)


def test_create_farhi_qaoa_circuits_when_number_of_layers_is_list():
    # Given
    hamiltonians = [
        QubitOperator("Z0 Z1"),
        QubitOperator("Z0") + QubitOperator("Z1"),
    ]
    number_of_layers = [2, 3]

    # When
    circuits = create_farhi_qaoa_circuits(hamiltonians, number_of_layers)

    # Then
    assert len(circuits) == len(hamiltonians)

    for circuit in circuits:
        assert isinstance(circuit, Circuit)


def test_create_farhi_qaoa_circuits_fails_when_length_of_inputs_is_not_equal():
    # Given
    hamiltonians = [
        QubitOperator("Z0 Z1"),
        QubitOperator("Z0") + QubitOperator("Z1"),
    ]
    number_of_layers = [2]

    # When
    with pytest.raises(AssertionError):
        create_farhi_qaoa_circuits(hamiltonians, number_of_layers)


def test_create_all_x_mixer_hamiltonian():
    # Given
    number_of_qubits = 4
    target_operator = (
        QubitOperator("X0")
        + QubitOperator("X1")
        + QubitOperator("X2")
        + QubitOperator("X3")
    )

    # When
    operator = create_all_x_mixer_hamiltonian(number_of_qubits)

    # Then
    assert operator == target_operator
