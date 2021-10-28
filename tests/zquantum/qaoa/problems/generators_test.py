from openfermion.utils import count_qubits
from zquantum.qaoa.problems import MaxCut, get_random_hamiltonians_for_problem


class TestGenerateRandomHamiltonians:
    def test_get_random_maxcut_hamiltonians_num_instances(self):
        # Given
        graph_specs = {"type_graph": "complete"}
        number_of_instances_list = [0, 1, 10]
        number_of_qubits = 4

        # When
        for number_of_instances in number_of_instances_list:
            hamiltonians = get_random_hamiltonians_for_problem(
                graph_specs,
                number_of_instances,
                number_of_qubits,
                MaxCut().get_hamiltonian,
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
            hamiltonians = get_random_hamiltonians_for_problem(
                graph_specs,
                number_of_instances,
                possible_number_of_qubits,
                MaxCut().get_hamiltonian,
            )

            # Then
            for hamiltonian in hamiltonians:
                assert count_qubits(hamiltonian) in possible_number_of_qubits
