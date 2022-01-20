import itertools

import pytest
from openfermion.utils import count_qubits
from zquantum.qaoa.problems import (
    MaxCut,
    get_random_hamiltonians_for_problem,
    get_random_ising_hamiltonian,
)


class TestGenerateRandomHamiltoniansForProblem:
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


class TestGetRandomHamiltonian:
    @pytest.mark.parametrize("num_qubits", [2, 5, 7])
    def test_random_hamiltonian_num_qubits(self, num_qubits):
        # Given
        if num_qubits >= 5:
            max_number_of_qubits_per_term = 4
        else:
            max_number_of_qubits_per_term = num_qubits

        # When
        hamiltonian = get_random_ising_hamiltonian(
            num_qubits, max_number_of_qubits_per_term
        )

        # Then
        assert count_qubits(hamiltonian) == num_qubits
        all_qubits_currently_in_hamiltonian = [
            term[0] for term in itertools.chain(*hamiltonian.terms.keys())
        ]
        for i in range(num_qubits):
            assert i in all_qubits_currently_in_hamiltonian

    @pytest.mark.parametrize("max_num_terms_per_qubit", [2, 4])
    def test_random_hamiltonian_max_num_terms_per_qubit(self, max_num_terms_per_qubit):
        # Given
        num_qubits = 5

        # When
        hamiltonian = get_random_ising_hamiltonian(num_qubits, max_num_terms_per_qubit)

        # Then
        for term in hamiltonian.terms.keys():
            assert len(term) <= max_num_terms_per_qubit
