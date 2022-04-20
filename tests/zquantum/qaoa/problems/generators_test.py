################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import pytest
from zquantum.core.openfermion import QubitOperator, change_operator_type
from zquantum.core.openfermion.utils import count_qubits
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


class TestGetRandomIsingHamiltonian:
    @pytest.mark.parametrize("num_terms", [2, 6])
    @pytest.mark.parametrize("num_qubits", [2, 5, 7])
    def test_num_qubits_and_num_terms_is_correct(self, num_qubits, num_terms):
        # Given
        if num_qubits >= 5:
            max_number_of_qubits_per_term = 4
        else:
            max_number_of_qubits_per_term = num_qubits

        # When
        hamiltonian = get_random_ising_hamiltonian(
            num_qubits, num_terms, max_number_of_qubits_per_term
        )

        # Then
        # Some qubits may not be included due to randomness, thus the generated number
        # of qubits may be less than `num_qubits`
        assert (
            count_qubits(change_operator_type(hamiltonian, QubitOperator)) <= num_qubits
        )
        generated_num_terms = len(hamiltonian.terms) - 1

        # If two of the randomly generated terms have the same qubits that are operated
        # on, then the two terms will be combined. Therefore, the generated number of
        # terms may be less than `num_terms`
        assert generated_num_terms <= num_terms

    @pytest.mark.parametrize("max_num_terms_per_qubit", [2, 4])
    def test_random_hamiltonian_max_num_terms_per_qubit(self, max_num_terms_per_qubit):
        # Given
        num_qubits = 5
        num_terms = 3

        # When
        hamiltonian = get_random_ising_hamiltonian(
            num_qubits, num_terms, max_num_terms_per_qubit
        )

        # Then
        for term in hamiltonian.terms.keys():
            assert len(term) <= max_num_terms_per_qubit
