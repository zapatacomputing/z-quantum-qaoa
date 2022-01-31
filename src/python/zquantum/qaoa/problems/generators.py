from typing import Callable, Dict, List

import networkx as nx
import numpy as np
from openfermion import QubitOperator
from zquantum.core.graph import generate_graph_from_specs


def get_random_hamiltonians_for_problem(
    graph_specs: Dict,
    number_of_instances: int,
    possible_number_of_qubits: List[int],
    hamiltonian_generator: Callable[[nx.Graph], QubitOperator],
    seed=None,
) -> List[QubitOperator]:
    """Generates Hamiltonians based on the input graph description for a range
    of number of qubits and a set number of instances.

    Args:
        graph_specs: Specifications of the graph to generate. It should contain at
            least an entry with key 'type_graph' (Note: 'num_nodes' key will be
            overwritten)
        number_of_instances: The number of hamiltonians to generate
        possible_number_of_qubits: A list containing the number of
            qubits in the hamiltonian. If it contains more than one value, then a
            random value from the list will be picked to generate each instance.
        hamiltonian_generator: a function that will generate a Hamiltonian
            for a given problem based on the input graph.
        seed: seed for random number generator.
    """
    if seed is not None:
        np.random.seed(seed)
    if "type_graph" not in graph_specs.keys():
        raise ValueError("graph_specs should contain type_graph field.")
    hamiltonians = []
    for _ in range(number_of_instances):
        graph_specs["num_nodes"] = np.random.choice(possible_number_of_qubits)
        graph = generate_graph_from_specs(graph_specs)

        hamiltonian = hamiltonian_generator(graph)
        hamiltonians.append(hamiltonian)

    return hamiltonians


def get_random_ising_hamiltonian(
    number_of_qubits: int, number_of_terms: int, max_number_of_qubits_per_term: int
) -> QubitOperator:
    """Generates a random Hamiltonian for a given number of qubits with weights
    between -1 and 1.

    Args:
        number_of_qubits: The number of qubits in the Hamiltonian. Should be >= 2.
        max_number_qubits_per_term: The maximum number of qubits for each term in the
            hamiltonian. Should be <= number_of_qubits.
    """

    hamiltonian = QubitOperator()
    for _ in range(number_of_terms):
        num_qubits_in_term = np.random.randint(1, max_number_of_qubits_per_term)
        qubits = np.random.choice(
            range(number_of_qubits), num_qubits_in_term, replace=False
        ).tolist()
        hamiltonian += QubitOperator(" ".join([f"Z{q}" for q in qubits])) * (
            np.random.rand() * 2 - 1
        )

    # Add constant term
    hamiltonian += QubitOperator("") * (np.random.rand() * 2 - 1)
    return hamiltonian
