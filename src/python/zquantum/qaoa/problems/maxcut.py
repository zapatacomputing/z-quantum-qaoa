from typing import Dict, List

import numpy as np
import networkx as nx
from openfermion import QubitOperator
from zquantum.core.utils import dec2bin
from zquantum.core.graph import generate_graph_node_dict, generate_graph_from_specs


def get_random_maxcut_hamiltonians(
    graph_specs: Dict,
    number_of_instances: int,
    possible_number_of_qubits: List[int],
    **kwargs
):
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

        hamiltonian = get_maxcut_hamiltonian(graph, **kwargs)
        hamiltonians.append(hamiltonian)

    return hamiltonians


def get_maxcut_hamiltonian(
    graph: nx.Graph,
    scaling: float = 1.0,
    shifted: bool = False,
    l1_normalized: bool = False,
) -> QubitOperator:
    """Converts a MAXCUT instance, as described by a weighted graph, to an Ising
    Hamiltonian. It allows for different convention in the choice of the
    Hamiltonian.

    Args:
        graph: undirected weighted graph describing the MAXCUT
        instance.
        scaling: scaling of the terms of the Hamiltonian
        shifted: if True include a shift. Default: False
        l1_normalized: normalize the operator using the l1_norm = \\sum |w|

    Returns:
        operator describing the Hamiltonian
        H = \\sum_{<i,j>} w_{i,j} * scaling * (Z_i Z_j - shifted * I)
        or H_norm = H / l1_norm if l1_normalized is True.

    """

    output = QubitOperator()

    nodes_dict = generate_graph_node_dict(graph)

    l1_norm = 0
    for edge in graph.edges:
        coeff = graph.edges[edge[0], edge[1]]["weight"] * scaling
        l1_norm += np.abs(coeff)
        node_index1 = nodes_dict[edge[0]]
        node_index2 = nodes_dict[edge[1]]
        ZZ_term_str = "Z" + str(node_index1) + " Z" + str(node_index2)
        output += QubitOperator(ZZ_term_str, coeff)
        if shifted:
            output += QubitOperator("", -coeff)  # constant term, i.e I
    if l1_normalized and (l1_norm > 0):
        output /= l1_norm
    return output


def get_solution_cut_size(solution, graph):
    """Compute the Cut given a partition of the nodes.

    Args:
        solution: list[0,1]
            A list of 0-1 values indicating the partition of the nodes of a graph into two
            separate sets.
        graph: networkx.Graph
            Input graph object.
    """

    if len(solution) != len(graph.nodes):
        raise Exception(
            "trial solution size is {}, which does not match graph size which is {}".format(
                len(solution), len(graph.nodes)
            )
        )

    cut_size = 0
    node_dict = generate_graph_node_dict(graph)
    for edge in graph.edges:
        node_index1 = node_dict[edge[0]]
        node_index2 = node_dict[edge[1]]
        if solution[node_index1] != solution[node_index2]:
            cut_size += 1
    return cut_size


def solve_maxcut_by_exhaustive_search(graph):
    """Brute-force solver for MAXCUT instances using exhaustive search.
    Args:
        graph (networkx.Graph): undirected weighted graph describing the MAXCUT
        instance.

    Returns:
        tuple: tuple whose first elements is the number of cuts, and second is a list
            of bit strings that correspond to the solution(s).
    """

    solution_set = []
    num_nodes = len(graph.nodes)

    # find one MAXCUT solution
    maxcut = -1
    one_maxcut_solution = None
    for i in range(0, 2 ** num_nodes):
        trial_solution = dec2bin(i, num_nodes)
        current_cut = get_solution_cut_size(trial_solution, graph)
        if current_cut > maxcut:
            one_maxcut_solution = trial_solution
            maxcut = current_cut
    solution_set.append(one_maxcut_solution)

    # search again to pick up any degeneracies
    for i in range(0, 2 ** num_nodes):
        trial_solution = dec2bin(i, num_nodes)
        current_cut = get_solution_cut_size(trial_solution, graph)
        if current_cut == maxcut and trial_solution != one_maxcut_solution:
            solution_set.append(trial_solution)

    return maxcut, solution_set
