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
) -> List[QubitOperator]:
    """Generates random maxcut hamiltonians based on the input graph description for a range
    of number of qubits and a set number of instances.

    Args:
        graph_specs (dict): Specifications of the graph to generate. It should contain at
            least an entry with key 'type_graph' (Note: 'num_nodes' key will be overwritten)
        number_of_instances (int): The number of hamiltonians to generate
        possible_number_of_qubits (List[int]): A list containing the number of
            qubits in the hamiltonian. If it contains more than one value, then a
            random value from the list will be picked to generate each instance.

    Returns:
        List of zquantum.core.qubitoperator.QubitOperator object describing the
        Hamiltonians
        H = \\sum_{<i,j>} w_{i,j} * scaling * (Z_i Z_j - shifted * I).

    """
    hamiltonians = []
    for _ in range(number_of_instances):
        graph_specs["num_nodes"] = np.random.choice(possible_number_of_qubits)
        graph = generate_graph_from_specs(graph_specs)

        hamiltonian = hamiltonian_generator(graph)
        hamiltonians.append(hamiltonian)

    return hamiltonians