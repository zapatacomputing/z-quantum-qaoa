import networkx as nx

from openfermion import QubitOperator
from openfermion.utils import count_qubits
from .maxcut import (
    get_maxcut_hamiltonian,
    get_solution_cut_size,
    solve_maxcut_by_exhaustive_search,
    get_random_maxcut_hamiltonians,
)


class TestMaxcut:
    def test_get_maxcut_hamiltonian(self):
        # Given
        graph = nx.Graph()
        graph.add_edge(1, 2, weight=0.4)
        graph.add_edge(2, 3, weight=-0.1)
        graph.add_edge(1, 3, weight=0.2)
        target_hamiltonian = (
            0.4 * QubitOperator("Z0 Z1")
            - 0.1 * QubitOperator("Z1 Z2")
            + 0.2 * QubitOperator("Z0 Z2")
        )

        # When
        hamiltonian = get_maxcut_hamiltonian(graph)

        # Then
        assert hamiltonian == target_hamiltonian

    def test_get_maxcut_hamiltonian_scaled_and_shifted(self):
        # Given
        graph = nx.Graph()
        graph.add_edge(1, 2, weight=0.4)
        graph.add_edge(2, 3, weight=-0.1)
        graph.add_edge(1, 3, weight=0.2)
        target_hamiltonian = (
            0.4 / 2 * QubitOperator("Z0 Z1")
            - 0.1 / 2 * QubitOperator("Z1 Z2")
            + 0.2 / 2 * QubitOperator("Z0 Z2")
            - (0.4 - 0.1 + 0.2) / 2 * QubitOperator("")
        )

        # When
        hamiltonian = get_maxcut_hamiltonian(graph, scaling=0.5, shifted=True)

        # Then
        assert hamiltonian == target_hamiltonian

    def test_get_maxcut_hamiltonian_l1normalized(self):
        # Given
        graph = nx.Graph()
        graph.add_edge(1, 2, weight=0.4)
        graph.add_edge(2, 3, weight=-0.1)
        graph.add_edge(1, 3, weight=0.2)
        target_hamiltonian = (
            0.4 / 0.7 * QubitOperator("Z0 Z1")
            - 0.1 / 0.7 * QubitOperator("Z1 Z2")
            + 0.2 / 0.7 * QubitOperator("Z0 Z2")
        )

        # When
        hamiltonian = get_maxcut_hamiltonian(graph, l1_normalized=True)

        # Then
        assert hamiltonian == target_hamiltonian

    def test_no_edge_l1normalized(self):
        # Given
        graph = nx.Graph()
        target_hamiltonian = QubitOperator()

        # When
        hamiltonian = get_maxcut_hamiltonian(graph, l1_normalized=True)

        # Then
        assert hamiltonian == target_hamiltonian

    def test_maxcut_exhaustive_solution(self):
        # Given
        graph = nx.Graph()
        graph.add_edge(1, 2, weight=1)
        graph.add_edge(1, 3, weight=1)
        graph.add_edge(2, 3, weight=1)
        graph.add_edge(2, 4, weight=1)
        graph.add_edge(3, 5, weight=1)
        graph.add_edge(4, 5, weight=1)
        # When
        maxcut, solution_set = solve_maxcut_by_exhaustive_search(graph)
        # Then
        assert maxcut == 5
        for solution in solution_set:
            cut_size = get_solution_cut_size(solution, graph)
            assert cut_size == 5

    def test_get_solution_cut_size(self):
        # Given
        solution_1 = [0, 0, 0, 0, 0]
        solution_2 = [0, 1, 1, 1, 1]
        solution_3 = [0, 0, 1, 0, 1]
        graph = nx.Graph()
        graph.add_edge(1, 2, weight=1)
        graph.add_edge(1, 3, weight=1)
        graph.add_edge(2, 3, weight=1)
        graph.add_edge(2, 4, weight=1)
        graph.add_edge(3, 5, weight=1)
        graph.add_edge(4, 5, weight=1)

        # When
        cut_size_1 = get_solution_cut_size(solution_1, graph)
        cut_size_2 = get_solution_cut_size(solution_2, graph)
        cut_size_3 = get_solution_cut_size(solution_3, graph)

        # Then
        assert cut_size_1 == 0
        assert cut_size_2 == 2
        assert cut_size_3 == 3

    def test_get_random_maxcut_hamiltonians_num_instances(self):
        # Given
        graph_specs = {"type_graph": "complete"}
        number_of_instances_list = [0, 1, 10]
        number_of_qubits = 4

        # When
        for number_of_instances in number_of_instances_list:
            hamiltonians = get_random_maxcut_hamiltonians(
                graph_specs, number_of_instances, number_of_qubits
            )

            # Then
            assert len(hamiltonians) == number_of_instances

    def test_get_random_maxcut_hamiltonians_num_qubits_is_in_range(self):
        # Given
        graph_specs = {"type_graph": "complete"}
        number_of_instances = 10
        list_possible_number_of_qubits = [[2, 3, 4], [2, 8]]

        # When
        for possible_number_of_qubits in list_possible_number_of_qubits:
            hamiltonians = get_random_maxcut_hamiltonians(
                graph_specs, number_of_instances, possible_number_of_qubits
            )

            # Then
            for hamiltonian in hamiltonians:
                assert count_qubits(hamiltonian) in possible_number_of_qubits
