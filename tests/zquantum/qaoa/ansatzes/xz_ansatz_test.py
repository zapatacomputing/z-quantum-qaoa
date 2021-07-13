from zquantum.core.interfaces.ansatz_test import AnsatzTests
from zquantum.core.circuits import Circuit, H, RZ, CNOT
from zquantum.core.utils import compare_unitary
from zquantum.core.openfermion import change_operator_type
from zquantum.qaoa.ansatzes.xz_ansatz import (
    QAOAXZAnsatz,
    create_xz_qaoa_circuits
)
from openfermion import QubitOperator, IsingOperator
import pytest
import sympy


def create_thetas(number_of_params):
    return sympy.symbols(f"theta_:{number_of_params}")

def create_symbols_map(number_of_params):
    symbols_map = {}
    thetas = create_thetas(number_of_params)
    for i in range(len(thetas)):
        symbols_map[thetas[i]] = 0.5
    return symbols_map

def create_target_unitary(number_of_params, k_body_depth = 1):
    thetas = create_thetas(number_of_params)
    symbols_map = create_symbols_map(number_of_params)

    target_circuit = Circuit()
    target_circuit += H(0)
    target_circuit += RZ(2 * thetas[0])(0)
    target_circuit += H(0)
    target_circuit += RZ(2 * thetas[1])(0)

    target_circuit += H(1)
    target_circuit += RZ(2 * thetas[2])(1)
    target_circuit += H(1)
    target_circuit += RZ(2 * thetas[3])(1)

    if k_body_depth == 2:
        target_circuit += H(0)
        target_circuit += H(1)
        target_circuit += CNOT(0, 1)
        target_circuit += RZ(0 * thetas[4])(1)
        target_circuit += CNOT(0, 1)
        target_circuit += H(1)
        target_circuit += H(0)

        target_circuit += CNOT(0, 1)
        target_circuit += RZ(0 * thetas[5])(1)
        target_circuit += CNOT(0, 1)

    return target_circuit.bind(symbols_map).to_unitary()

def create_target_unitary_type_2(number_of_params, k_body_depth = 1):
    thetas = create_thetas(number_of_params)
    symbols_map = create_symbols_map(number_of_params)

    target_circuit = Circuit()
    target_circuit += H(0)
    target_circuit += RZ(2 * thetas[0])(0)
    target_circuit += H(0)
    target_circuit += RZ(2 * thetas[1])(0)
    target_circuit += RZ(2 * thetas[1])(1)

    target_circuit += H(1)
    target_circuit += RZ(2 * thetas[2])(1)
    target_circuit += H(1)
    target_circuit += RZ(2 * thetas[3])(0)
    target_circuit += RZ(2 * thetas[3])(1)

    if k_body_depth == 2:
        target_circuit += H(0)
        target_circuit += H(1)
        target_circuit += CNOT(0, 1)
        target_circuit += RZ(0 * thetas[4])(1)
        target_circuit += CNOT(0, 1)
        target_circuit += H(1)
        target_circuit += H(0)

        target_circuit += RZ(0 * thetas[5])(0)
        target_circuit += RZ(0 * thetas[5])(1)

    return target_circuit.bind(symbols_map).to_unitary()

class TestQAOAXAnsatz(AnsatzTests):
    @pytest.fixture()
    def number_of_params(self):
        return 6

    @pytest.fixture
    def ansatz(self):
        cost_hamiltonian = QubitOperator(("Z0 Z1")) + QubitOperator(("Z1 Z2"))
        return QAOAXZAnsatz(
            number_of_layers=1,
            cost_hamiltonian=cost_hamiltonian,
            type=1,
        )

    def test_get_number_of_qubits(self, ansatz):
        # Given
        cost_hamiltonian = QubitOperator(("Z0 Z1")) + QubitOperator(("Z1 Z2"))
        target_number_of_qubits = 3

        # When
        ansatz.cost_hamiltonian = cost_hamiltonian

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

    def test_get_number_of_params(self, ansatz):
        # Given
        cost_hamiltonian = QubitOperator(("Z0 Z1")) + QubitOperator(("Z1 Z2"))
        target_number_of_params = 6

        # When
        ansatz.cost_hamiltonian = cost_hamiltonian

        # Then
        assert ansatz.number_of_params == target_number_of_params

    def test_get_number_of_params_with_ising_hamiltonian(self, ansatz):
        # Given
        new_cost_hamiltonian = (
            QubitOperator((0, "Z")) + QubitOperator((1, "Z")) + QubitOperator((2, "Z"))
        )
        new_cost_hamiltonian = change_operator_type(new_cost_hamiltonian, IsingOperator)
        target_number_of_params = 6

        # When
        ansatz.cost_hamiltonian = new_cost_hamiltonian

        # Then
        assert ansatz.number_of_params == target_number_of_params

    def test_get_number_of_params_with_k_body_depth_greater_than_1(self, ansatz):
        # Given
        cost_hamiltonian = QubitOperator(("Z0 Z1")) + QubitOperator(("Z1 Z2"))
        target_number_of_params = 12

        # When
        ansatz.cost_hamiltonian = cost_hamiltonian
        ansatz.number_of_layers = 2

        # Then
        assert ansatz.number_of_params == target_number_of_params

    def test_generate_circuit(self, ansatz):
        # Given
        symbols_map = create_symbols_map(number_of_params=4)
        target_unitary = create_target_unitary(number_of_params=4)

        ansatz.cost_hamiltonian = QubitOperator(("Z0 Z1"))

        # When
        parametrized_circuit = ansatz._generate_circuit()
        evaluated_circuit = parametrized_circuit.bind(symbols_map)
        final_unitary = evaluated_circuit.to_unitary()

        # Then
        assert compare_unitary(final_unitary, target_unitary, tol=1e-10)

    def test_generate_circuit_with_k_body_depth_greater_than_1(self, ansatz):
        # Given
        symbols_map = create_symbols_map(number_of_params=6)
        target_unitary = create_target_unitary(number_of_params=6, k_body_depth=2)

        ansatz.number_of_layers = 2
        ansatz.cost_hamiltonian = QubitOperator(("Z0 Z1"))

        # When
        parametrized_circuit = ansatz._generate_circuit()
        evaluated_circuit = parametrized_circuit.bind(symbols_map)
        final_unitary = evaluated_circuit.to_unitary()

        # Then
        assert compare_unitary(final_unitary, target_unitary, tol=1e-10)

    def test_generate_circuit_type_2(self, ansatz):
        # Given
        symbols_map = create_symbols_map(number_of_params=4)
        target_unitary = create_target_unitary_type_2(number_of_params=4)

        ansatz.type = 1
        ansatz.cost_hamiltonian = QubitOperator(("Z0 Z1"))

        # When
        parametrized_circuit = ansatz._generate_circuit()
        evaluated_circuit = parametrized_circuit.bind(symbols_map)
        final_unitary = evaluated_circuit.to_unitary()
        # Then
        assert compare_unitary(final_unitary, target_unitary, tol=1e-10)

    def test_generate_circuit_type_2_with_k_body_depth_greater_than_1(self, ansatz):
        # Given
        symbols_map = create_symbols_map(number_of_params=6)
        target_unitary = create_target_unitary_type_2(number_of_params=6, k_body_depth=2)
        ansatz.number_of_layers = 2
        ansatz.type = 1
        ansatz.cost_hamiltonian = QubitOperator(("Z0 Z1"))
        
        # When
        parametrized_circuit = ansatz._generate_circuit()
        evaluated_circuit = parametrized_circuit.bind(symbols_map)
        final_unitary = evaluated_circuit.to_unitary()

        import pdb
        pdb.set_trace()
        # Then
        assert compare_unitary(final_unitary, target_unitary, tol=1e-10)


def test_create_x_qaoa_circuits():
    # Given
    hamiltonians = [
        QubitOperator("Z0 Z1"),
        QubitOperator("Z0") + QubitOperator("Z1"),
    ]
    number_of_layers = 2

    # When
    circuits = create_xz_qaoa_circuits(hamiltonians, number_of_layers)

    # Then
    assert len(circuits) == len(hamiltonians)

    for circuit in circuits:
        assert isinstance(circuit, Circuit)


def test_create_x_qaoa_circuits_fails_when_length_of_inputs_is_not_equal():
    # Given
    hamiltonians = [
        QubitOperator("Z0 Z1"),
        QubitOperator("Z0") + QubitOperator("Z1"),
    ]
    number_of_layers = [2]

    # When
    with pytest.raises(AssertionError):
        create_xz_qaoa_circuits(hamiltonians, number_of_layers)
