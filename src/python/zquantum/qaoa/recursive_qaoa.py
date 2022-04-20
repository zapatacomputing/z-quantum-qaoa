################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
from collections import defaultdict
from copy import copy, deepcopy
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.optimize import OptimizeResult
from zquantum.core.history.recorder import HistoryEntry
from zquantum.core.history.recorder import recorder as _recorder
from zquantum.core.interfaces.ansatz import Ansatz
from zquantum.core.interfaces.cost_function import CostFunction
from zquantum.core.interfaces.optimizer import (
    NestedOptimizer,
    Optimizer,
    extend_histories,
    optimization_result,
)
from zquantum.core.openfermion import IsingOperator, QubitOperator, change_operator_type
from zquantum.core.openfermion.utils import count_qubits
from zquantum.core.typing import RecorderFactory
from zquantum.qaoa.problems import solve_problem_by_exhaustive_search


class RecursiveQAOA(NestedOptimizer):
    @property
    def inner_optimizer(self) -> Optimizer:
        return self._inner_optimizer

    @property
    def recorder(self) -> RecorderFactory:
        return self._recorder

    def __init__(
        self,
        n_c: int,
        cost_hamiltonian: IsingOperator,
        ansatz: Ansatz,
        inner_optimizer: Optimizer,
        recorder: RecorderFactory = _recorder,
    ) -> None:
        """Recursive QAOA (RQAOA) optimizer

        Original paper: https://arxiv.org/abs/1910.08980 page 4.

        The main idea is that we call QAOA recursively and reduce the size of the cost
        hamiltonian by 1 on each recursion, until we hit a threshold number of qubits
        `n_c`. Then, we use brute force to solve the reduced QAOA problem, mapping the
        reduced solution to the original solution.

        Args:
            n_c: The threshold number of qubits at which recursion stops, as described
                in the original paper. Cannot be greater than number of qubits.
            cost_hamiltonian: Hamiltonian representing the cost function.
            ansatz: an Ansatz object with all params (ex. `n_layers`) initialized
            inner_optimizer: optimizer used for optimization of parameters at each
                recursion of RQAOA.
            recorder: recorder object defining how to store the optimization history.
        """
        n_qubits = count_qubits(change_operator_type(cost_hamiltonian, QubitOperator))

        if n_c >= n_qubits or n_c <= 0:
            raise ValueError(
                "n_c needs to be a value less than number of qubits and greater than 0."
            )

        self._n_c = n_c
        self._ansatz = ansatz
        self._cost_hamiltonian = cost_hamiltonian
        self._inner_optimizer = inner_optimizer
        self._recorder = recorder

    def _minimize(
        self,
        cost_function_factory: Callable[[IsingOperator, Ansatz], CostFunction],
        initial_params: np.ndarray,
        keep_history: bool = False,
    ) -> OptimizeResult:
        """Args:
            cost_function_factory: function that generates CostFunction objects given
                the provided ansatz and cost_hamiltonian.
            initial_params: initial parameters used for optimization
            keep_history: flag indicating whether history of cost function evaluations
                should be recorded.

        Returns:
            OptimizeResult with the added entry of:
                opt_solutions (List[Tuple[int, ...]]): The solution(s) to recursive
                    QAOA as a list of tuples; each tuple is a tuple of bits.
        """

        n_qubits = count_qubits(
            change_operator_type(self._cost_hamiltonian, QubitOperator)
        )
        qubit_map = _create_default_qubit_map(n_qubits)

        histories: Dict[str, List[HistoryEntry]] = defaultdict(list)
        histories["history"] = []

        return self._recursive_minimize(
            cost_function_factory,
            initial_params,
            keep_history,
            cost_hamiltonian=self._cost_hamiltonian,
            qubit_map=qubit_map,
            nit=0,
            nfev=0,
            histories=histories,
        )

    def _recursive_minimize(
        self,
        cost_function_factory,
        initial_params,
        keep_history,
        cost_hamiltonian,
        qubit_map,
        nit,
        nfev,
        histories,
    ):
        """A method that recursively calls itself with each recursion reducing 1 term
        of the cost hamiltonian
        """

        # Set up QAOA circuit
        ansatz = copy(self._ansatz)

        ansatz.cost_hamiltonian = cost_hamiltonian

        cost_function = cost_function_factory(cost_hamiltonian, ansatz)

        if keep_history:
            cost_function = self.recorder(cost_function)

        # Run & optimize QAOA
        opt_results = self.inner_optimizer.minimize(cost_function, initial_params)
        nit += opt_results.nit
        nfev += opt_results.nfev
        if keep_history:
            histories = extend_histories(cost_function, histories)

        # Reduce the cost hamiltonian
        (
            term_with_largest_expval,
            largest_expval,
        ) = _find_term_with_strongest_correlation(
            cost_hamiltonian,
            ansatz,
            opt_results.opt_params,
            cost_function_factory,
        )

        new_qubit_map = _update_qubit_map(
            qubit_map, term_with_largest_expval, largest_expval
        )

        reduced_cost_hamiltonian = _create_reduced_hamiltonian(
            cost_hamiltonian,
            term_with_largest_expval,
            largest_expval,
        )

        # Check new cost hamiltonian has correct amount of qubits
        assert (
            count_qubits(change_operator_type(reduced_cost_hamiltonian, QubitOperator))
            == count_qubits(change_operator_type(cost_hamiltonian, QubitOperator)) - 1
            # If we have 1 qubit, the reduced cost hamiltonian would be empty and say
            # it has 0 qubits.
            or count_qubits(
                change_operator_type(reduced_cost_hamiltonian, QubitOperator)
            )
            == 0
            and count_qubits(change_operator_type(cost_hamiltonian, QubitOperator)) == 2
            and self._n_c == 1
        )

        # Check qubit map has correct amount of qubits
        assert (
            count_qubits(change_operator_type(cost_hamiltonian, QubitOperator)) - 1
            == max([qubit_indices[0] for qubit_indices in new_qubit_map.values()]) + 1
        )

        if (
            count_qubits(change_operator_type(reduced_cost_hamiltonian, QubitOperator))
            > self._n_c
        ):
            # If we didn't reach threshold `n_c`, we repeat the the above with the
            # reduced cost hamiltonian.
            return self._recursive_minimize(
                cost_function_factory,
                initial_params,
                keep_history,
                cost_hamiltonian=reduced_cost_hamiltonian,
                qubit_map=new_qubit_map,
                nit=nit,
                nfev=nfev,
                histories=histories,
            )

        else:
            best_value, reduced_solutions = solve_problem_by_exhaustive_search(
                change_operator_type(reduced_cost_hamiltonian, QubitOperator)
            )

            solutions = _map_reduced_solutions_to_original_solutions(
                reduced_solutions, new_qubit_map
            )

            opt_result = optimization_result(
                opt_solutions=solutions,
                opt_value=best_value,
                opt_params=None,
                nit=nit,
                nfev=nfev,
                **histories,
            )

            return opt_result


def _create_default_qubit_map(n_qubits: int) -> Dict[int, List[int]]:
    """Creates a qubit map that maps each qubit to itself."""
    qubit_map = {}
    for i in range(n_qubits):
        qubit_map[i] = [i, 1]
    return qubit_map


def _find_term_with_strongest_correlation(
    hamiltonian: IsingOperator,
    ansatz: Ansatz,
    optimal_params: np.ndarray,
    cost_function_factory: Callable[[IsingOperator, Ansatz], CostFunction],
) -> Tuple[IsingOperator, float]:
    """Find term Z_i Z_j maximizing <psi(beta, gamma) | Z_i Z_j | psi(beta, gamma)>.

    The idea is that the term with largest expectation value also has the largest
    correlation or anticorrelation between its qubits, and this information can be used
    to eliminate a qubit. See equation (15) of the original paper.

    Args:
        hamiltonian: the hamiltonian that you want to find term with strongest
            correlation of.
        ansatz: ansatz representing the circuit of the full hamiltonian, used to
            calculate psi(beta, gamma)
        optimal_params: optimal values of beta, gamma
        cost_function_factory: See docstring of RecursiveQAOA

    Returns:
        The term with the largest correlation, and its expectation value.
    """
    largest_expval = 0.0

    for term in hamiltonian:
        # If term is a constant term, don't calculate expectation value.
        if () not in term.terms:

            # Calculate expectation value of term
            cost_function_of_term = cost_function_factory(term, ansatz)
            expval_of_term = cost_function_of_term(optimal_params)  # type: ignore

            if np.abs(expval_of_term) > np.abs(largest_expval):
                largest_expval = expval_of_term
                term_with_largest_expval = term

    return (term_with_largest_expval, largest_expval)


def _update_qubit_map(
    qubit_map: Dict[int, List[int]],
    term_with_largest_expval: IsingOperator,
    largest_expval: float,
) -> Dict[int, List[int]]:
    """Update the qubit map according to equation (15) in the original paper.

    The process comprises the following steps:
        1. Substituting one qubit of `term_with_largest_expval` with the other
        2. Substituting all qubits larger than the gotten-rid-of-qubit with the qubit
           one below it

    Args:
        qubit_map: the qubit map to be updated.
        term_with_largest_expval: term with largest expectation value
        largest_expval: the expectation value of `term_with_largest_expval`

    Note:
        Qubit map is a dictionary that maps qubits in reduced Hamiltonian back to
        original qubits.

        Example:
            `qubit_map = {0: [2, -1], 1: [3, 1]]}
                Keys are the original qubit indices.
                1st term of inner list is qubit the index of tuple to be mapped onto,
                2nd term is if it will be mapped onto the same value or opposite of
                    the qubit it is being mapped onto.
                In the above qubit_map, the original qubit 0 is now represented by the
                    opposite value of qubit 2, and the original qubit 1 is now
                    represented by the value of qubit 3.
    """
    assert len(term_with_largest_expval.terms.keys()) == 1

    new_qubit_map = deepcopy(qubit_map)

    qubit_to_get_rid_of: int = [*term_with_largest_expval.terms][0][1][0]

    # i is original qubit, qubit_map[i][0] is current equivalent of original qubit.
    for i in range(len(new_qubit_map)):
        if new_qubit_map[i][0] == qubit_to_get_rid_of:
            new_qubit_map[i][1] *= int(np.sign(largest_expval))
        new_qubit_map[i][0] = _get_new_qubit_indice(
            new_qubit_map[i][0], term_with_largest_expval
        )

    return new_qubit_map


def _get_new_qubit_indice(
    old_indice: int, operator_with_largest_expval: IsingOperator
) -> int:
    assert len(operator_with_largest_expval.terms.keys()) == 1

    term_with_largest_expval = [*operator_with_largest_expval.terms][0]
    # term_with_largest_expval is now a subscriptable tuple like ((0, 'Z'), (1, 'Z'))

    qubit_to_get_rid_of: int = term_with_largest_expval[1][0]
    qubit_itll_be_replaced_with: int = term_with_largest_expval[0][0]

    new_indice = old_indice

    if old_indice > qubit_to_get_rid_of:
        # map qubit to the qubit 1 below it
        new_indice = old_indice - 1
    elif old_indice == qubit_to_get_rid_of:
        # map qubit onto the qubit it's being replaced with
        new_indice = qubit_itll_be_replaced_with

    return new_indice


def _create_reduced_hamiltonian(
    hamiltonian: IsingOperator,
    term_with_largest_expval: IsingOperator,
    largest_expval: float,
) -> IsingOperator:
    """Reduce the cost hamiltonian accordinmg to eq. (15) in the original paper.

    Reduction is done by substituting one qubit of the term with the largest
    expectation value with the other qubit of the term. See equation (15) of the
    original paper.

    Args:
        hamiltonian: hamiltonian to be reduced
        term_with_largest_expval: term with largest expectation value
        largest_expval: the expectation value of `term_with_largest_expval`

    Returns:
        Reduced hamiltonian.
    """
    assert len(term_with_largest_expval.terms.keys()) == 1

    reduced_hamiltonian = IsingOperator()

    qubit_to_get_rid_of: int = [*term_with_largest_expval.terms][0][1][0]

    for (term, coefficient) in hamiltonian.terms.items():
        # term is tuple representing one term of IsingOperator, example
        # ((2, 'Z'), (3, 'Z'))
        if term not in term_with_largest_expval.terms:
            # If term is not the term_with_largest_expval
            new_term: Tuple = ()
            for qubit in term:
                # qubit is a component of qubit operator on 1 qubit ex. (2, 'Z')
                qubit_indice: int = qubit[0]

                # Map the new cost hamiltonian onto reduced qubits
                new_qubit_indice = _get_new_qubit_indice(
                    qubit_indice, term_with_largest_expval
                )
                new_qubit = (new_qubit_indice, "Z")
                new_term += (new_qubit,)

                if qubit_indice == qubit_to_get_rid_of:
                    coefficient *= np.sign(largest_expval)

            reduced_hamiltonian += IsingOperator(new_term, coefficient)

    return reduced_hamiltonian


def _map_reduced_solutions_to_original_solutions(
    reduced_solutions: List[Tuple[int]], qubit_map: Dict[int, List[int]]
):
    """Map the answer of the reduced Hamiltonian back to the original number of qubits.

    Args:
        reduced_solutions: list of solutions, each solution is a tuple of ints.
        qubit_map: list that maps original qubits to new qubits, see docstring of
            _update_qubit_map for more details.

    Returns:
        list of solutions, each solution is a tuple of ints.
    """

    original_solutions: List[Tuple[int, ...]] = []

    for reduced_solution in reduced_solutions:
        original_solution: List[int] = []
        for qubit, sign in qubit_map.values():
            this_answer = reduced_solution[qubit]

            # If negative, flip the qubit.
            if sign == -1:
                if this_answer == 0:
                    this_answer = 1
                else:
                    this_answer = 0
            original_solution.append(this_answer)

        original_solutions.append(tuple(original_solution))

    return original_solutions
