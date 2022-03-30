import pytest
from zquantum.core.openfermion import QubitOperator
from zquantum.qaoa.problems import solve_problem_by_exhaustive_search

HAMILTONIAN_SOLUTION_COST_LIST = [
    (
        QubitOperator("Z0 Z1") + QubitOperator("[]", -1),
        [(0, 1), (1, 0)],
        -2,
    ),
    (
        QubitOperator("Z0 Z1", 5)
        + QubitOperator("Z0 Z3", 5)
        + QubitOperator("Z1 Z2", 0.5)
        + QubitOperator("Z2 Z3", 0.5)
        + QubitOperator("[]", -11),
        [(0, 1, 0, 1), (1, 0, 1, 0)],
        -22,
    ),
]


class TestSolveProblemByExhaustiveSearch:
    @pytest.mark.parametrize(
        "hamiltonian,target_solutions,target_value", [*HAMILTONIAN_SOLUTION_COST_LIST]
    )
    def test_solve_problem_by_exhaustive_search(
        self, hamiltonian, target_solutions, target_value
    ):
        value, solutions = solve_problem_by_exhaustive_search(hamiltonian)
        assert set(solutions) == set(target_solutions)
        assert value == target_value
