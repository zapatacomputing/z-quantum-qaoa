from functools import partial

import numpy as np
import pytest
from openfermion import IsingOperator
from zquantum.core.cost_function import (
    create_cost_function,
    substitution_based_estimation_tasks_factory,
)
from zquantum.core.estimation import (
    allocate_shots_uniformly,
    estimate_expectation_values_by_averaging,
)
from zquantum.core.interfaces.cost_function import CostFunction
from zquantum.core.interfaces.mock_objects import MockOptimizer
from zquantum.core.interfaces.optimizer import optimization_result
from zquantum.core.symbolic_simulator import SymbolicSimulator
from zquantum.qaoa.ansatzes.farhi_ansatz import QAOAFarhiAnsatz
from zquantum.qaoa.recursive_qaoa import (
    RecursiveQAOA,
    _create_default_qubit_map,
    _create_reduced_hamiltonian,
    _find_term_with_strongest_correlation,
    _map_reduced_solutions_to_original_solutions,
    _update_qubit_map,
)


class TestRQAOA:
    @pytest.fixture()
    def hamiltonian(self):
        return (
            IsingOperator("Z0 Z1", 5)
            + IsingOperator("Z0 Z3", 2)
            + IsingOperator("Z1 Z2", 0.5)
            + IsingOperator("Z2 Z3", 0.5)
        )

    @pytest.fixture()
    def ansatz(self, hamiltonian):
        return QAOAFarhiAnsatz(1, hamiltonian)

    @pytest.fixture()
    def estimation_tasks_factory_generator(self):
        estimation_preprocessors = [
            partial(allocate_shots_uniformly, number_of_shots=1000)
        ]
        return partial(
            substitution_based_estimation_tasks_factory,
            estimation_preprocessors=estimation_preprocessors,
        )

    @pytest.fixture()
    def cost_function_factory(self):
        return partial(
            create_cost_function,
            backend=SymbolicSimulator(),
            estimation_method=estimate_expectation_values_by_averaging,
            parameter_preprocessors=None,
        )

    @pytest.fixture()
    def opt_params(self):
        # Pre-figured-out optimal params for the hamiltonian fixture
        return np.array([-0.34897497, 4.17835486])

    @pytest.fixture()
    def optimizer(self, opt_params):
        optimizer = MockOptimizer()

        def custom_minimize(
            cost_function: CostFunction,
            initial_params: np.ndarray,
            keep_history: bool = False,
        ):
            return optimization_result(
                opt_value=cost_function(opt_params),
                opt_params=opt_params,
                history=[],
            )

        optimizer._minimize = custom_minimize
        return optimizer

    @pytest.mark.parametrize(
        "n_qubits, expected_qubit_map",
        [
            (2, [[0, 1], [1, 1]]),
            (7, [[0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1]]),
        ],
    )
    def test_create_default_qubit_map(self, n_qubits, expected_qubit_map):
        qubit_map = _create_default_qubit_map(n_qubits)
        assert qubit_map == expected_qubit_map

    def test_find_term_with_strongest_correlation(
        self,
        hamiltonian,
        ansatz,
        opt_params,
        estimation_tasks_factory_generator,
        cost_function_factory,
    ):

        (
            term_with_largest_expval,
            largest_expval,
        ) = _find_term_with_strongest_correlation(
            hamiltonian,
            ansatz,
            opt_params,
            estimation_tasks_factory_generator,
            cost_function_factory,
        )
        assert term_with_largest_expval == IsingOperator("Z0 Z1", 5)
        assert np.sign(largest_expval) == -1

    @pytest.mark.parametrize(
        "term_with_largest_expval, largest_expval, expected_reduced_ham, expected_new_qubit_map",
        [
            (
                # suppose we want to get rid of qubit 1 and replace it with qubit 0.
                IsingOperator("Z0 Z1", 5.0),
                10,
                (
                    IsingOperator("Z0 Z2", 2.0)
                    + IsingOperator("Z0 Z1", 0.5)
                    + IsingOperator("Z1 Z2", 0.5)
                ),
                [[0, 1], [0, 1], [1, 1], [2, 1]],
            ),
            (
                IsingOperator("Z0 Z1", 5.0),
                -10,
                (
                    IsingOperator("Z0 Z2", 2.0)
                    + IsingOperator("Z0 Z1", -0.5)
                    + IsingOperator("Z1 Z2", 0.5)
                ),
                [[0, 1], [0, -1], [1, 1], [2, 1]],
            ),
            (
                IsingOperator("Z0 Z3", 2),
                -10,
                (
                    IsingOperator("Z0 Z1", 5)
                    + IsingOperator("Z1 Z2", 0.5)
                    + IsingOperator("Z2 Z0", -0.5)
                ),
                [[0, 1], [1, 1], [2, 1], [0, -1]],
            ),
        ],
    )
    def test_reduce_cost_hamiltonian_and_qubit_map(
        self,
        hamiltonian,
        term_with_largest_expval,
        largest_expval,
        expected_reduced_ham,
        expected_new_qubit_map,
    ):
        qubit_map = _create_default_qubit_map(4)

        new_qubit_map = _update_qubit_map(
            qubit_map, term_with_largest_expval, largest_expval
        )
        reduced_ham = _create_reduced_hamiltonian(
            hamiltonian, term_with_largest_expval, largest_expval, qubit_map
        )
        assert reduced_ham.terms == expected_reduced_ham.terms
        assert new_qubit_map == expected_new_qubit_map

    def test_update_qubit_map_works_properly_on_subsequent_recursions(self):
        # (This test is for when the qubit map to be updated is not the default qubit map)
        qubit_map = [[0, 1], [1, 1], [1, -1], [2, 1], [1, 1]]
        term_with_largest_expval = IsingOperator("Z0 Z1")
        largest_expval = -42

        # How the expected_new_qubit_map is calculated:
        # [[0, 1], [1, 1], [1, -1], [2, 1], [1, 1]] -> original qubit map
        # [[0, 1], [0, -1], [1, -1], [2, 1], [1, 1]] -> replace 1 with negative of 0
        # [[0, 1], [0, -1], [0, 1], [2, 1], [0, -1]] -> replace things that depends on 1 with negative of 0
        # [[0, 1], [0, -1], [0, 1], [1, 1], [0, -1]] -> nudge higher qubits down
        expected_new_qubit_map = [[0, 1], [0, -1], [0, 1], [1, 1], [0, -1]]
        new_qubit_map = _update_qubit_map(
            qubit_map, term_with_largest_expval, largest_expval
        )
        assert new_qubit_map == expected_new_qubit_map

    @pytest.mark.parametrize(
        "qubit_map, expected_original_solutions",
        [
            ([[0, 1], [1, 1]], [(0, 1), (1, 0)]),
            ([[1, 1], [1, 1], [0, -1]], [(1, 1, 1), (0, 0, 0)]),
            (
                [[0, 1], [0, -1], [0, 1], [1, 1], [0, -1]],
                [(0, 1, 0, 1, 1), (1, 0, 1, 0, 0)],
            ),
        ],
    )
    def test_map_reduced_solutions_to_original_solutions(
        self, qubit_map, expected_original_solutions
    ):
        reduced_solutions = [(0, 1), (1, 0)]
        original_solutions = _map_reduced_solutions_to_original_solutions(
            reduced_solutions, qubit_map
        )

        # Check lists are equal regardless of order
        assert set(expected_original_solutions) == set(original_solutions)

    def test_RQAOA_returns_correct_answer(
        self,
        hamiltonian,
        optimizer,
        ansatz,
        estimation_tasks_factory_generator,
        cost_function_factory,
    ):
        initial_params = np.array([0.42, 4.2])

        recursive_qaoa = RecursiveQAOA(
            3,
            ansatz,
            initial_params,
            optimizer,
            estimation_tasks_factory_generator,
            cost_function_factory,
        )

        solutions = recursive_qaoa(hamiltonian)

        n_qubits = 4
        for solution in solutions:
            assert len(solution) == n_qubits

        breakpoint()

        assert set(solutions) == set([(1, 0, 1, 0), (0, 1, 0, 1)])


# TODO: Test that recursions are correct.
