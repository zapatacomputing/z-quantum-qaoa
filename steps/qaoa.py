from zquantum.qaoa.maxcut import (
    get_random_maxcut_hamiltonians as _get_random_maxcut_hamiltonians,
    create_farhi_qaoa_circuits as _create_farhi_qaoa_circuits,
    get_maxcut_hamiltonian as _get_maxcut_hamiltonian,
    solve_maxcut_by_exhaustive_search as _solve_maxcut_by_exhaustive_search,
)
from zquantum.core.circuit import save_circuit, save_circuit_set
from zquantum.core.graph import load_graph
from qeopenfermion import (
    save_qubit_operator_set,
    load_qubit_operator,
    load_qubit_operator_set,
    save_qubit_operator,
)
import json


def get_random_maxcut_hamiltonians(
    graph_specs, number_of_instances, number_of_qubits, shifted=False, scaling=1.0
):
    graph_specs_dict = json.loads(graph_specs)
    hamiltonians = _get_random_maxcut_hamiltonians(
        graph_specs_dict,
        number_of_instances,
        number_of_qubits,
        shifted=shifted,
        scaling=scaling,
    )
    save_qubit_operator_set(hamiltonians, "hamiltonians.json")


def create_farhi_qaoa_circuit(number_of_layers, hamiltonian):
    hamiltonian_object = load_qubit_operator(hamiltonian)
    circuit = _create_farhi_qaoa_circuits([hamiltonian_object], number_of_layers)[0]
    save_circuit(circuit, "circuit.json")


def create_farhi_qaoa_circuits(number_of_layers, hamiltonians):
    hamiltonians_objects = load_qubit_operator_set(hamiltonians)
    circuits = _create_farhi_qaoa_circuits(hamiltonians_objects, number_of_layers)
    save_circuit_set(circuits, "circuits.json")


def get_maxcut_hamiltonian(graph, scaling=1.0, shifted=False):
    graph_object = load_graph(graph)
    hamiltonian = _get_maxcut_hamiltonian(
        graph_object, scaling=scaling, shifted=shifted
    )
    save_qubit_operator(hamiltonian, "hamiltonian.json")
