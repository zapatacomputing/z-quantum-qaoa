from zquantum.core.interfaces.ansatz_test import AnsatzTests
from zquantum.core.circuits import Circuit, H, RZ, CNOT
from zquantum.core.utils import compare_unitary
from zquantum.core.openfermion import change_operator_type
from zquantum.qaoa.ansatzes.x_ansatz import (
    QAOAXAnsatz,
    create_x_qaoa_circuits,
    cost_of_cut,
    get_edges_from_cost_hamiltonian,
)
from openfermion import QubitOperator, IsingOperator
import pytest
import sympy
from zquantum.qaoa.problems.maxcut import get_maxcut_hamiltonian
import networkx as nx


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
    target_circuit += H(1)
    target_circuit += RZ(4 * thetas[1])(1)
    target_circuit += H(1)
    target_circuit += H(2)
    target_circuit += RZ(2 * thetas[2])(2)
    target_circuit += H(2)

    if k_body_depth == 2:
        target_circuit += H(0)
        target_circuit += H(1)
        target_circuit += CNOT(0, 1)
        target_circuit += RZ(2 * thetas[3])(1)
        target_circuit += CNOT(0, 1)
        target_circuit += H(1)
        target_circuit += H(0)

        target_circuit += H(0)
        target_circuit += H(2)
        target_circuit += CNOT(0, 2)
        target_circuit += RZ(4 * thetas[4])(2)
        target_circuit += CNOT(0, 2)
        target_circuit += H(0)
        target_circuit += H(2)

        target_circuit += H(1)
        target_circuit += H(2)
        target_circuit += CNOT(1, 2)
        target_circuit += RZ(2 * thetas[5])(2)
        target_circuit += CNOT(1, 2)
        target_circuit += H(2)
        target_circuit += H(1)

    return target_circuit.bind(symbols_map).to_unitary()


class TestQAOAXAnsatz(AnsatzTests):
    @pytest.fixture()
    def number_of_params(self):
        return 3

    @pytest.fixture
    def ansatz(self):
        cost_hamiltonian = QubitOperator(("Z0 Z1")) + QubitOperator(("Z1 Z2"))
        return QAOAXAnsatz(
            number_of_layers=1,
            cost_hamiltonian=cost_hamiltonian,
        )

    @pytest.fixture
    def target_unitary(self, number_of_params):
        return create_target_unitary(number_of_params)

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
        target_number_of_params = 3

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
        target_number_of_params = 3

        # When
        ansatz.cost_hamiltonian = new_cost_hamiltonian

        # Then
        assert ansatz.number_of_params == target_number_of_params

    def test_get_number_of_params_with_k_body_depth_greater_than_1(self, ansatz):
        # Given
        cost_hamiltonian = QubitOperator(("Z0 Z1")) + QubitOperator(("Z1 Z2"))
        target_number_of_params = 6

        # When
        ansatz.cost_hamiltonian = cost_hamiltonian
        ansatz.number_of_layers = 2

        # Then
        assert ansatz.number_of_params == target_number_of_params

    def test_get_edges_from_cost_hamiltonian(self):
        # When
        cost_hamiltonian = QubitOperator(("Z0 Z1")) + QubitOperator(("Z1 Z2"))
        expected_edges = [[0, 1, 2.0], [1, 2, 2.0]]
        final_edges = get_edges_from_cost_hamiltonian(cost_hamiltonian)

        # Then
        assert expected_edges == final_edges

    def test_get_edges_with_ising_hamiltonian(self):
        # When
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        G.add_edge(0, 1, weight=5)
        G.add_edge(1, 2, weight=2)
        cost_hamiltonian = get_maxcut_hamiltonian(G)
        expected_edges = [[1, 2, 2.0], [0, 1, 5.0]]
        final_edges = get_edges_from_cost_hamiltonian(cost_hamiltonian)

        # Then
        assert expected_edges == final_edges

    def test_get_cost_of_cut(self):
        # When
        edges = [[0, 1, 2.0], [1, 2, 2.0]]
        cut = (0,)
        expected_cost = 2.0
        final_cost = cost_of_cut(cut, edges)

        # Then
        assert expected_cost == final_cost        

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
        target_unitary = create_target_unitary(number_of_params=6, k_body_depth = 2)
        ansatz.number_of_layers = 2
        parametrized_circuit = ansatz._generate_circuit()
        evaluated_circuit = parametrized_circuit.bind(symbols_map)
        final_unitary = evaluated_circuit.to_unitary()

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
    circuits = create_x_qaoa_circuits(hamiltonians, number_of_layers)

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
        create_x_qaoa_circuits(hamiltonians, number_of_layers)
