from zquantum.core.interfaces.ansatz_test import AnsatzTests
from zquantum.core.circuit import Circuit, Gate, Qubit
from zquantum.core.utils import compare_unitary
from zquantum.core.openfermion import change_operator_type
from .warm_start_ansatz import WarmStartQAOAAnsatz, convert_relaxed_solution_to_angles
from openfermion import QubitOperator, IsingOperator
import pytest
import numpy as np
import sympy


def create_betas(number_of_layers):
    return sympy.symbols(f"beta_:{number_of_layers}")


def create_gammas(number_of_layers):
    return sympy.symbols(f"gamma_:{number_of_layers}")


def create_symbols_map(number_of_layers):
    betas = create_betas(number_of_layers)
    gammas = create_gammas(number_of_layers)
    symbols_map = []
    beta_value = 0.5
    gamma_value = 0.7
    for beta, gamma in zip(betas, gammas):
        symbols_map.append([beta, beta_value])
        symbols_map.append([gamma, gamma_value])
    return symbols_map


def create_target_unitary(thetas, number_of_layers):
    target_circuit = Circuit()
    target_circuit.gates = []
    target_circuit.gates.append(Gate("Ry", [Qubit(0)], [thetas[0]]))
    target_circuit.gates.append(Gate("Ry", [Qubit(1)], [thetas[1]]))
    betas = create_betas(number_of_layers)
    gammas = create_gammas(number_of_layers)
    symbols_map = create_symbols_map(number_of_layers)
    for layer_id in range(number_of_layers):
        beta = betas[layer_id]
        gamma = gammas[layer_id]
        target_circuit.gates.append(Gate("Rz", [Qubit(0)], [2.0 * gamma]))
        target_circuit.gates.append(Gate("Rz", [Qubit(1)], [2.0 * gamma]))
        target_circuit.gates.append(Gate("Ry", [Qubit(0)], [-thetas[0]]))
        target_circuit.gates.append(Gate("Ry", [Qubit(1)], [-thetas[1]]))
        target_circuit.gates.append(Gate("Rz", [Qubit(0)], [-2.0 * beta]))
        target_circuit.gates.append(Gate("Rz", [Qubit(1)], [-2.0 * beta]))
        target_circuit.gates.append(Gate("Ry", [Qubit(0)], [thetas[0]]))
        target_circuit.gates.append(Gate("Ry", [Qubit(1)], [thetas[1]]))

    return target_circuit.evaluate(symbols_map).to_unitary()


class TestWarmStartQAOAAnsatz(AnsatzTests):
    @pytest.fixture
    def thetas(self):
        return np.array([0.5, 0.5])

    @pytest.fixture(params=[1, 2, 3])
    def number_of_layers(self, request):
        return request.param

    @pytest.fixture
    def ansatz(self, thetas, number_of_layers):
        cost_hamiltonian = QubitOperator((0, "Z")) + QubitOperator((1, "Z"))
        return WarmStartQAOAAnsatz(
            number_of_layers=number_of_layers,
            cost_hamiltonian=cost_hamiltonian,
            thetas=thetas,
        )

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

    def test_get_parametrizable_circuit(self, ansatz, number_of_layers):
        # Then
        betas = create_betas(number_of_layers)
        gammas = create_gammas(number_of_layers)
        target_params = []
        for i in range(number_of_layers):
            target_params.append(gammas[i])
            target_params.append(betas[i])

        assert ansatz.parametrized_circuit.symbolic_params == target_params

    def test_generate_circuit(self, ansatz, number_of_layers, thetas):
        # When
        symbols_map = create_symbols_map(number_of_layers)
        parametrized_circuit = ansatz._generate_circuit()
        evaluated_circuit = parametrized_circuit.evaluate(symbols_map)
        final_unitary = evaluated_circuit.to_unitary()
        target_unitary = create_target_unitary(thetas, number_of_layers)
        # Then
        assert compare_unitary(final_unitary, target_unitary, tol=1e-10)

    def test_generate_circuit_with_ising_operator(
        self, ansatz, number_of_layers, thetas
    ):
        # When
        ansatz.cost_hamiltonian = change_operator_type(
            ansatz.cost_hamiltonian, IsingOperator
        )

        parametrized_circuit = ansatz._generate_circuit()
        symbols_map = create_symbols_map(number_of_layers)
        target_unitary = create_target_unitary(thetas, number_of_layers)
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
