import pytest
import sympy
from zquantum.core.circuits import CNOT, RZ, Circuit, H
from zquantum.core.interfaces.ansatz_test import AnsatzTests
from zquantum.core.utils import compare_unitary
from zquantum.qaoa.ansatzes.x_ansatz import XAnsatz, XZAnsatz


def create_thetas(number_of_params):
    return sympy.symbols(f"theta_:{number_of_params}")


def create_symbols_map(number_of_params):
    symbols_map = {}
    thetas = create_thetas(number_of_params)
    for i in range(len(thetas)):
        symbols_map[thetas[i]] = 0.5
    return symbols_map


def create_x_operator(qubit: int, param):
    return Circuit([H(qubit), RZ(2 * param)(qubit), H(qubit)])


def create_2_qubit_x_operator(qubit1: int, qubit2: int, param):
    return Circuit(
        [
            H(qubit1),
            H(qubit2),
            CNOT(qubit1, qubit2),
            RZ(2 * param)(qubit2),
            CNOT(qubit1, qubit2),
            H(qubit1),
            H(qubit2),
        ]
    )


def create_X_target_unitary(number_of_params, k_body_depth=1):
    thetas = create_thetas(number_of_params)
    symbols_map = create_symbols_map(number_of_params)

    target_circuit = Circuit()
    # Add an x operator to every qubit
    for i in range(3):
        target_circuit += create_x_operator(i, thetas[i])

    if k_body_depth == 2:
        target_circuit += create_2_qubit_x_operator(0, 1, thetas[3])
        target_circuit += create_2_qubit_x_operator(0, 2, thetas[4])
        target_circuit += create_2_qubit_x_operator(1, 2, thetas[5])

    return target_circuit.bind(symbols_map).to_unitary()


def create_XZ1_target_unitary(number_of_params, k_body_depth=1):
    thetas = create_thetas(number_of_params)
    symbols_map = create_symbols_map(number_of_params)

    target_circuit = Circuit()
    target_circuit += create_x_operator(0, thetas[0])
    target_circuit += RZ(2 * thetas[1])(0)

    target_circuit += create_x_operator(1, thetas[2])
    target_circuit += RZ(2 * thetas[3])(1)

    if k_body_depth == 2:
        target_circuit += create_2_qubit_x_operator(0, 1, thetas[4])
        target_circuit += CNOT(0, 1)
        target_circuit += RZ(2 * thetas[5])(1)
        target_circuit += CNOT(0, 1)

    return target_circuit.bind(symbols_map).to_unitary()


def create_XZ2_target_unitary(number_of_params, k_body_depth=1):
    thetas = create_thetas(number_of_params)
    symbols_map = create_symbols_map(number_of_params)

    target_circuit = Circuit()
    target_circuit += create_x_operator(0, thetas[0])
    target_circuit += RZ(2 * thetas[1])(0)
    target_circuit += RZ(2 * thetas[1])(1)

    target_circuit += create_x_operator(1, thetas[2])
    target_circuit += RZ(2 * thetas[3])(0)
    target_circuit += RZ(2 * thetas[3])(1)

    if k_body_depth == 2:
        target_circuit += create_2_qubit_x_operator(0, 1, thetas[4])

        target_circuit += RZ(2 * thetas[5])(0)
        target_circuit += RZ(2 * thetas[5])(1)

    return target_circuit.bind(symbols_map).to_unitary()


class TestXAnsatz(AnsatzTests):
    @pytest.fixture()
    def number_of_params(self):
        return 3

    @pytest.fixture
    def ansatz(self):
        return XAnsatz(
            number_of_layers=1,
            number_of_qubits=3,
        )

    @pytest.fixture
    def target_unitary(self, number_of_params):
        return create_X_target_unitary(number_of_params)

    def test_generate_circuit(self, ansatz, number_of_params, target_unitary):
        # When
        symbols_map = create_symbols_map(number_of_params)
        parametrized_circuit = ansatz._generate_circuit()
        evaluated_circuit = parametrized_circuit.bind(symbols_map)
        final_unitary = evaluated_circuit.to_unitary()

        # Then
        assert compare_unitary(final_unitary, target_unitary, tol=1e-10)

    def test_generate_circuit_with_k_body_depth_greater_than_1(self, ansatz):
        # When
        symbols_map = create_symbols_map(number_of_params=6)
        target_unitary = create_X_target_unitary(number_of_params=6, k_body_depth=2)
        ansatz.number_of_layers = 2
        parametrized_circuit = ansatz._generate_circuit()
        evaluated_circuit = parametrized_circuit.bind(symbols_map)
        final_unitary = evaluated_circuit.to_unitary()

        # Then
        assert compare_unitary(final_unitary, target_unitary, tol=1e-10)


class TestXZAnsatz(AnsatzTests):
    @pytest.fixture()
    def number_of_params(self):
        return 4

    @pytest.fixture
    def ansatz(self):
        return XZAnsatz(
            number_of_layers=1,
            number_of_qubits=2,
        )

    def test_generate_circuit(self, ansatz):
        # Given
        symbols_map = create_symbols_map(number_of_params=4)
        target_unitary = create_XZ1_target_unitary(number_of_params=4)

        # When
        parametrized_circuit = ansatz._generate_circuit()
        evaluated_circuit = parametrized_circuit.bind(symbols_map)
        final_unitary = evaluated_circuit.to_unitary()

        # Then
        assert compare_unitary(final_unitary, target_unitary, tol=1e-10)

    def test_generate_circuit_with_k_body_depth_greater_than_1(self, ansatz):
        # Given
        symbols_map = create_symbols_map(number_of_params=6)
        target_unitary = create_XZ1_target_unitary(number_of_params=6, k_body_depth=2)

        ansatz.number_of_layers = 2

        # When
        parametrized_circuit = ansatz._generate_circuit()
        evaluated_circuit = parametrized_circuit.bind(symbols_map)
        final_unitary = evaluated_circuit.to_unitary()

        # Then
        assert compare_unitary(final_unitary, target_unitary, tol=1e-10)

    def test_generate_circuit_type_2(self, ansatz):
        # Given
        symbols_map = create_symbols_map(number_of_params=4)
        target_unitary = create_XZ2_target_unitary(number_of_params=4)

        ansatz.use_k_body_z_operators = False

        # When
        parametrized_circuit = ansatz._generate_circuit()
        evaluated_circuit = parametrized_circuit.bind(symbols_map)
        final_unitary = evaluated_circuit.to_unitary()
        # Then
        assert compare_unitary(final_unitary, target_unitary, tol=1e-10)

    def test_generate_circuit_type_2_with_k_body_depth_greater_than_1(self, ansatz):
        # Given
        symbols_map = create_symbols_map(number_of_params=6)
        target_unitary = create_XZ2_target_unitary(number_of_params=6, k_body_depth=2)
        ansatz.number_of_layers = 2
        ansatz.use_k_body_z_operators = False

        # When
        parametrized_circuit = ansatz._generate_circuit()
        evaluated_circuit = parametrized_circuit.bind(symbols_map)
        final_unitary = evaluated_circuit.to_unitary()

        # Then
        assert compare_unitary(final_unitary, target_unitary, tol=1e-10)
