from zquantum.core.interfaces.ansatz_test import AnsatzTests
from zquantum.core.circuit import Circuit, Gate, Qubit
from zquantum.core.utils import compare_unitary
from zquantum.core.openfermion import change_operator_type
from .warm_start_ansatz import WarmStartQAOAAnsatz, convert_relaxed_solution_to_angles
from openfermion import QubitOperator, IsingOperator
import pytest
import numpy as np
import sympy


class TestWarmStartQAOAAnsatz(AnsatzTests):
    @pytest.fixture
    def thetas(self):
        return np.array([0.5, 0.5])

    @pytest.fixture
    def ansatz(self, thetas):
        cost_hamiltonian = QubitOperator((0, "Z")) + QubitOperator((1, "Z"))
        return WarmStartQAOAAnsatz(
            number_of_layers=1,
            cost_hamiltonian=cost_hamiltonian,
            thetas=thetas,
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
    def target_unitary(self, beta, gamma, thetas, symbols_map):
        target_circuit = Circuit()
        target_circuit.gates = []
        target_circuit.gates.append(Gate("Ry", [Qubit(0)], [thetas[0]]))
        target_circuit.gates.append(Gate("Ry", [Qubit(1)], [thetas[1]]))
        target_circuit.gates.append(Gate("Rz", [Qubit(0)], [2.0 * gamma]))
        target_circuit.gates.append(Gate("Rz", [Qubit(1)], [2.0 * gamma]))
        target_circuit.gates.append(Gate("Ry", [Qubit(0)], [-thetas[0]]))
        target_circuit.gates.append(Gate("Ry", [Qubit(1)], [-thetas[1]]))
        target_circuit.gates.append(Gate("Rz", [Qubit(0)], [-2.0 * beta]))
        target_circuit.gates.append(Gate("Rz", [Qubit(1)], [-2.0 * beta]))
        target_circuit.gates.append(Gate("Ry", [Qubit(0)], [thetas[0]]))
        target_circuit.gates.append(Gate("Ry", [Qubit(1)], [thetas[1]]))

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

    def test_set_thetas(self, ansatz):
        # Given
        new_thetas = np.array([0.3, 0.3])

        # When
        ansatz.thetas = new_thetas

        # Then
        ansatz._thetas == new_thetas

    def test_set_thetas_invalidates_circuit(self, ansatz):
        # Given
        new_thetas = np.array([0.3, 0.3])

        # When
        ansatz.thetas = new_thetas

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


def test_convert_relaxed_solution_to_angles():
    relaxed_solution = np.array([0, 0.5, 1])
    epsilon = 0.1
    target_converted_solution = np.array(
        [
            2 * np.arcsin(np.sqrt(epsilon)),
            2 * np.arcsin(np.sqrt(0.5)),
            2 * np.arcsin(np.sqrt(1 - epsilon)),
        ]
    )
    converted_solution = convert_relaxed_solution_to_angles(
        relaxed_solution, epsilon=epsilon
    )
    assert np.allclose(converted_solution, target_converted_solution)


def test_convert_relaxed_solution_to_angles_throws_exception_for_invalid_parameters():
    relaxed_solution = np.array([-1, 2, 1])

    with pytest.raises(ValueError):
        converted_solution = convert_relaxed_solution_to_angles(
            relaxed_solution, epsilon=0.1
        )
