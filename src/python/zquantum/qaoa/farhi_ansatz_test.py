from zquantum.core.interfaces.ansatz_test import AnsatzTests
from zquantum.core.circuit import Circuit, Gate, Qubit
from zquantum.core.utils import compare_unitary
from .farhi_ansatz import QAOAFarhiAnsatz
from openfermion import QubitOperator
import unittest
import numpy as np
import sympy


class TestQAOAFarhiAnsatz(unittest.TestCase, AnsatzTests):
    def setUp(self):
        self.number_of_layers = 1
        self.cost_hamiltonian = QubitOperator((0, "Z")) + QubitOperator((1, "Z"))
        self.mixer_hamiltonian = QubitOperator((0, "X")) + QubitOperator((1, "X"))

        self.ansatz = QAOAFarhiAnsatz(
            number_of_layers=self.number_of_layers,
            cost_hamiltonian=self.cost_hamiltonian,
            mixer_hamiltonian=self.mixer_hamiltonian,
        )

        self.beta = sympy.Symbol("beta_0")
        self.gamma = sympy.Symbol("gamma_0")
        beta_value = 0.5
        gamma_value = 0.7
        self.symbols_map = [(self.beta, beta_value), (self.gamma, gamma_value)]
        self.target_circuit = Circuit()
        self.target_circuit.gates = []
        self.target_circuit.gates.append(Gate("H", [Qubit(0)]))
        self.target_circuit.gates.append(Gate("H", [Qubit(1)]))
        self.target_circuit.gates.append(Gate("Rz", [Qubit(0)], [2.0 * self.gamma]))
        self.target_circuit.gates.append(Gate("Rz", [Qubit(1)], [2.0 * self.gamma]))
        self.target_circuit.gates.append(Gate("Rx", [Qubit(0)], [2.0 * self.beta]))
        self.target_circuit.gates.append(Gate("Rx", [Qubit(1)], [2.0 * self.beta]))
        self.target_unitary = self.target_circuit.evaluate(
            self.symbols_map
        ).to_unitary()

    def test_set_cost_hamiltonian(self):
        # Given
        new_cost_hamiltonian = QubitOperator((0, "Z")) - QubitOperator((1, "Z"))

        # When
        self.ansatz.cost_hamiltonian = new_cost_hamiltonian

        # Then
        self.assertEqual(self.ansatz._cost_hamiltonian, new_cost_hamiltonian)

    def test_set_cost_hamiltonian_invalidates_circuit(self):
        # Given
        new_cost_hamiltonian = QubitOperator((0, "Z")) - QubitOperator((1, "Z"))

        # When
        self.ansatz.cost_hamiltonian = new_cost_hamiltonian

        # Then
        self.assertIsNone(self.ansatz._parametrized_circuit)

    def test_set_mixer_hamiltonian(self):
        # Given
        new_mixer_hamiltonian = QubitOperator((0, "Z")) - QubitOperator((1, "Z"))

        # When
        self.ansatz.mixer_hamiltonian = new_mixer_hamiltonian

        # Then
        self.assertEqual(self.ansatz._mixer_hamiltonian, new_mixer_hamiltonian)

    def test_set_mixer_hamiltonian_invalidates_circuit(self):
        # Given
        new_mixer_hamiltonian = QubitOperator((0, "Z")) - QubitOperator((1, "Z"))

        # When
        self.ansatz.mixer_hamiltonian = new_mixer_hamiltonian

        # Then
        self.assertIsNone(self.ansatz._parametrized_circuit)

    def test_get_number_of_qubits(self):
        # Given
        new_cost_hamiltonian = (
            QubitOperator((0, "Z")) + QubitOperator((1, "Z")) + QubitOperator((2, "Z"))
        )
        target_number_of_qubits = 3

        # When
        self.ansatz.cost_hamiltonian = new_cost_hamiltonian

        # Then
        self.assertEqual(self.ansatz.number_of_qubits, target_number_of_qubits)

    def test_get_parametrizable_circuit(self):
        # When
        parametrized_circuit = self.ansatz.parametrized_circuit

        # Then
        self.assertEqual(
            parametrized_circuit.symbolic_params, set([self.beta, self.gamma])
        )

    def test_generate_circuit(self):
        # When
        parametrized_circuit = self.ansatz._generate_circuit()
        evaluated_circuit = parametrized_circuit.evaluate(self.symbols_map)
        final_unitary = evaluated_circuit.to_unitary()

        # Then
        self.assertTrue(compare_unitary(final_unitary, self.target_unitary, tol=1e-10))

    def test_get_symbols(self):
        # Given
        target_symbols = [
            sympy.Symbol("gamma_0"),
            sympy.Symbol("beta_0"),
            sympy.Symbol("gamma_1"),
            sympy.Symbol("beta_1"),

        ]
        self.ansatz.number_of_layers = 2

        # When
        symbols = self.ansatz.get_symbols()

        # When/Then
        self.assertEqual(symbols, target_symbols)
