################################################################################
# © Copyright 2020-2022 Zapata Computing Inc.
################################################################################
import json
from typing import List, Union

import numpy as np
from zquantum.core.circuits import save_circuit, save_circuitset
from zquantum.core.graph import load_graph
from zquantum.core.openfermion import (
    load_qubit_operator,
    load_qubit_operator_set,
    save_qubit_operator,
    save_qubit_operator_set,
)
from zquantum.core.utils import load_list, save_list
from zquantum.qaoa.ansatzes import farhi_ansatz, warm_start_ansatz
from zquantum.qaoa.problems import (
    GraphPartitioning,
    MaxCut,
    get_random_hamiltonians_for_problem,
)


def get_random_maxcut_hamiltonians(graph_specs, number_of_instances, number_of_qubits):
    graph_specs_dict = json.loads(graph_specs)
    hamiltonians = get_random_hamiltonians_for_problem(
        graph_specs_dict,
        number_of_instances,
        number_of_qubits,
        MaxCut().get_hamiltonian,
    )
    save_qubit_operator_set(hamiltonians, "hamiltonians.json")


def create_farhi_qaoa_circuit(number_of_layers, hamiltonian):
    hamiltonian_object = load_qubit_operator(hamiltonian)
    circuit = farhi_ansatz.create_farhi_qaoa_circuits(
        [hamiltonian_object], number_of_layers
    )[0]
    save_circuit(circuit, "circuit.json")


def create_farhi_qaoa_circuits(
    number_of_layers: Union[int, List[int], str], hamiltonians
):
    if isinstance(number_of_layers, str):
        number_of_layers = load_list(number_of_layers)
    hamiltonians_objects = load_qubit_operator_set(hamiltonians)
    circuits = farhi_ansatz.create_farhi_qaoa_circuits(
        hamiltonians_objects, number_of_layers
    )
    save_circuitset(circuits, "circuits.json")


def get_maxcut_hamiltonian(graph, scale_factor=1.0, offset=0.0):
    graph_object = load_graph(graph)
    hamiltonian = MaxCut().get_hamiltonian(
        graph_object, scale_factor=scale_factor, offset=offset
    )
    save_qubit_operator(hamiltonian, "hamiltonian.json")


def get_graph_partition_hamiltonian(graph, scale_factor=1.0, offset=0.0):
    graph_object = load_graph(graph)
    hamiltonian = GraphPartitioning().get_hamiltonian(
        graph_object, scale_factor=scale_factor, offset=offset
    )
    save_qubit_operator(hamiltonian, "hamiltonian.json")


def convert_relaxed_solution_to_angles(solution, epsilon=0.5):
    solution = np.array(load_list(solution))
    thetas = warm_start_ansatz.convert_relaxed_solution_to_angles(solution, epsilon)
    save_list(thetas.tolist(), "thetas.json")
