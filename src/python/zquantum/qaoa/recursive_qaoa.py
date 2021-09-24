import abc
from copy import copy, deepcopy
from typing import Callable, Dict, List, Tuple

import numpy as np
from openfermion import IsingOperator, QubitOperator
from openfermion.utils import count_qubits
from typing_extensions import Protocol
from zquantum.core.interfaces.ansatz import Ansatz
from zquantum.core.interfaces.cost_function import CostFunction
from zquantum.core.estimation import EstimationTasksFactory
from zquantum.core.interfaces.optimizer import Optimizer
from zquantum.core.openfermion import change_operator_type
from zquantum.qaoa.problems import solve_problem_by_exhaustive_search


class CostFunctionFactory(Protocol):
    @abc.abstractmethod
    def __call__(
        self, estimation_tasks_factory: EstimationTasksFactory
    ) -> CostFunction:
        """Creates a cost function from an EstimationTasksFactory object."""


class RecursiveQAOA:
    def __init__(
        self,
        n_c: int,
        ansatz: Ansatz,
        initial_params: np.ndarray,
        optimizer: Optimizer,
        estimation_tasks_factory: Callable[
            [IsingOperator, Ansatz], EstimationTasksFactory
        ],
        cost_function_factory: CostFunctionFactory,
    ) -> None:
        """This is an implementation of recursive QAOA (RQAOA) from https://arxiv.org/abs/1910.08980 page 4.

        The main idea is that we call QAOA recursively and reduce the size of the cost hamiltonian by 1
        on each recursion, until we hit a threshold number of qubits `n_c`. Then, we use brute force to
        solve the reduced QAOA problem, mapping the reduced solution to the original solution.

        Args:
            n_c: The threshold number of qubits at which recursion stops, as described in the original paper.
                Cannot be greater than number of qubits.
            ansatz: an Ansatz object with all params (ex. `n_layers`) initialized
            initial_params: initial parameters used for optimization
            optimizer: Optimizer object used for optimizer
            estimation_tasks_factory_generator: function that generates EstimationTasksFactory objects
                from operator and ansatz. See example below for clarification.
            cost_function_factory: function that generates CostFunction objects given EstimationTasksFactory.
                See example below for clarification.

        Example usage (aka, what the heck are all these factories?):

            from functools import partial
            from zquantum.core.estimation import (
                estimate_expectation_values_by_averaging,
                allocate_shots_uniformly
            )
            from zquantum.core.cost_function import (
                substitution_based_estimation_tasks_factory,
                create_cost_function,
            )

            cost_hamiltonian = ...
            ansatz = ...

            estimation_preprocessors = [partial(allocate_shots_uniformly, number_of_shots=1000)]
            estimation_tasks_factory_generator = partial(
                substitution_based_estimation_tasks_factory,
                estimation_preprocessors=estimation_preprocessors
            )
            cost_function_factory = partial(
                create_cost_function,
                backend=QuantumBackend,
                estimation_method=estimate_expectation_values_by_averaging,
                parameter_preprocessors=None,
            )

            initial_params = np.array([0.42, 4.2])
            optimizer = ...

            recursive_qaoa = RecursiveQAOA(
                3,
                ansatz,
                initial_params,
                optimizer,
                estimation_tasks_factory_generator,
                cost_function_factory,
            )

            solutions = recursive_qaoa(cost_hamiltonian)
        """

        self._n_c = n_c
        self._ansatz = ansatz
        self._initial_params = initial_params
        self._optimizer = optimizer
        self._estimation_tasks_factory = estimation_tasks_factory
        self._cost_function_factory = cost_function_factory

    def __call__(
        self,
        cost_hamiltonian: IsingOperator,
        qubit_map: Dict[int, List[int]] = None,
    ) -> List[Tuple[int, ...]]:
        """Args:
            cost_hamiltonian: Hamiltonian representing the cost function.
            qubit_map: A dictionary that maps qubits in reduced Hamiltonian back to original qubits, used
                for subsequent recursions. (Not for the first recursion.)
                Example:
                    `qubit_map = {0: [2, -1], 1: [3, 1]]}
                        Keys are the original qubit indice.
                        1st term of inner list is qubit the index of tuple to be mapped onto,
                        2nd term is if it will be mapped onto the same value or opposite of the qubit it
                            is being mapped onto.
                        In the above qubit_map, the original qubit 0 is now represented by the opposite
                            value of qubit 2, and the original qubit 1 is now represented by the value of
                            qubit 3.

        Returns:
            The solution(s) to recursive QAOA as a list of tuples; each tuple is a tuple of bits.
        """

        n_qubits = count_qubits(change_operator_type(cost_hamiltonian, QubitOperator))

        if self._n_c >= n_qubits or self._n_c <= 0:
            raise ValueError(
                "n_c needs to be a value less than number of qubits and greater than 0."
            )

        if qubit_map is None:
            qubit_map = _create_default_qubit_map(n_qubits)

        ansatz = copy(self._ansatz)

        if hasattr(ansatz, "_cost_hamiltonian"):
            ansatz._cost_hamiltonian = cost_hamiltonian
        else:
            # X ansatzes (zquantum.qaoa.ansatz.XAnsatz) generate based on number of qubits
            # instead of cost hamiltonian
            Warning(
                "Ansatz does not have a `_cost_hamiltonian` attribute, so `number_of_qubits` will be used to generate circuits."
            )
            ansatz.number_of_qubits = n_qubits

        estimation_tasks_factory = self._estimation_tasks_factory(
            cost_hamiltonian, ansatz
        )
        cost_function = self._cost_function_factory(
            estimation_tasks_factory=estimation_tasks_factory,
        )

        # Run & optimize QAOA
        opt_results = self._optimizer.minimize(cost_function, self._initial_params)

        (
            term_with_largest_expval,
            largest_expval,
        ) = _find_term_with_strongest_correlation(
            cost_hamiltonian,
            ansatz,
            opt_results.opt_params,
            self._estimation_tasks_factory,
            self._cost_function_factory,
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
            # If we have 1 qubit, the reduced cost hamiltonian would be empty and say it has
            # 0 qubits.
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
            == max([l[0] for l in new_qubit_map.values()]) + 1
        )

        if (
            count_qubits(change_operator_type(reduced_cost_hamiltonian, QubitOperator))
            > self._n_c
        ):
            # If we didn't reach threshold `n_c`, we repeat the the above with the reduced
            # cost hamiltonian.
            return self.__call__(reduced_cost_hamiltonian, new_qubit_map)

        else:
            best_value, reduced_solutions = solve_problem_by_exhaustive_search(
                change_operator_type(reduced_cost_hamiltonian, QubitOperator)
            )

            return _map_reduced_solutions_to_original_solutions(
                reduced_solutions, new_qubit_map
            )


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
    estimation_tasks_factory: Callable[[IsingOperator, Ansatz], EstimationTasksFactory],
    cost_function_factory: CostFunctionFactory,
) -> Tuple[IsingOperator, float]:
    """For each term Z_i Z_j, calculate the expectation value <psi(beta, gamma) | Z_i Z_j | psi(beta, gamma)>
    with optimal beta and gamma. The idea is that the term with largest expectation value
    has the largest correlation or anticorrelation between its qubits, and this information
    can be used to eliminate a qubit. See equation (15) of the original paper.

    Args:
        hamiltonian: the hamiltonian that you want to find term with strongest correlation of.
        ansatz: ansatz representing the circuit of the full hamiltonian, used to calculate psi(beta, gamma)
        optimal_params: optimal values of beta, gamma
        estimation_tasks_factory: See docstring of RecursiveQAOA
        cost_function_factory: See docstring of RecursiveQAOA

    Returns:
        The term with the largest correlation, and the value of that term's expectation value.
    """
    largest_expval = 0.0

    for term in hamiltonian:
        # If term is a constant term, don't calculate expectation value.
        if () not in term.terms:

            # Calculate expectation value of term
            estimation_tasks_factory_of_term = estimation_tasks_factory(term, ansatz)
            cost_function_of_term = cost_function_factory(
                estimation_tasks_factory=estimation_tasks_factory_of_term
            )
            expval_of_term = cost_function_of_term(optimal_params)

            if np.abs(expval_of_term) > np.abs(largest_expval):
                largest_expval = expval_of_term
                term_with_largest_expval = term

    return (term_with_largest_expval, largest_expval)


def _update_qubit_map(
    qubit_map: Dict[int, List[int]],
    term_with_largest_expval: IsingOperator,
    largest_expval: float,
) -> Dict[int, List[int]]:
    """Updates the qubit map by
        1. Substituting one qubit of `term_with_largest_expval` with the other
        2. Substituting all qubits larger than the gotten-rid-of-qubit with the qubit one below it
    See equation (15) of the original paper.

    Args:
        qubit_map: the qubit map to be updated. a list that maps original qubits to new qubits,
            see docstring of RecursiveQAOA
        term_with_largest_expval: term with largest expectation value
        largest_expval: the expectation value of `term_with_largest_expval`

    Returns:
        Updated qubit map

    """
    assert len(term_with_largest_expval.terms.keys()) == 1

    new_qubit_map = deepcopy(qubit_map)

    qubit_to_get_rid_of: int = [*term_with_largest_expval.terms][0][1][0]

    # i is original qubit, qubit_map[i][0] is current qubit equivalent of original qubit.
    for i in range(len(new_qubit_map)):
        if new_qubit_map[i][0] == qubit_to_get_rid_of:
            new_qubit_map[i][1] *= int(np.sign(largest_expval))
        new_qubit_map[i][0] = _get_new_qubit_indice(
            new_qubit_map[i][0], term_with_largest_expval
        )

    return new_qubit_map


def _get_new_qubit_indice(
    old_indice: int, term_with_largest_expval: IsingOperator
) -> int:
    assert len(term_with_largest_expval.terms.keys()) == 1

    term_with_largest_expval = [*term_with_largest_expval.terms][0]
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
    """Reduce the cost hamiltonian by substituting one qubit of the term with largest expectation
    value with the other qubit of the term. See equation (15) of the original paper.

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
        # term is tuple representing one term of IsingOperator, example ((2, 'Z'), (3, 'Z'))
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
    """Maps the answer of the reduced Hamiltonian back to the original number of qubits.

    Args:
        reduced_solutions: list of solutions, each solution is a tuple of ints.
        qubit_map: list that maps original qubits to new qubits, see docstring of RecursiveQAOA

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
