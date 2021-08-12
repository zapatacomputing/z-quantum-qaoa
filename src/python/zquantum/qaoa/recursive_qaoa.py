from zquantum.core.interfaces.cost_function import EstimationTasksFactory, CostFunction
from zquantum.core.interfaces.ansatz import Ansatz
from zquantum.core.interfaces.optimizer import Optimizer
from zquantum.core.openfermion import change_operator_type
from openfermion import QubitOperator, IsingOperator
from openfermion.utils import count_qubits
import numpy as np
from typing import Callable, Optional, List, Tuple
from zquantum.qaoa.problems import solve_problem_by_exhaustive_search
from copy import copy


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
        cost_function_factory: Callable[[EstimationTasksFactory], CostFunction],
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
                estimation_tasks_factory,
                cost_function_factory,
            )

            solutions = RecursiveQAOA(cost_hamiltonian)
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
        qubit_map: Optional[List[List[int]]] = None,
    ) -> List[Tuple]:
        """Args:
            cost_hamiltonian: Hamiltonian representing the cost function.
            qubit_map: A list that maps qubits in reduced Hamiltonian back to original qubits, used for
                subsequent recursions. (Not for the first recursion.)
                Example:
                    `qubit_map = [(2, -1), (3, 1)]`
                        Indice of each tuple is the original qubit indice.
                        1st term of tuple is qubit the index of tuple to be mapped onto,
                        2nd term is if it will be mapped onto the same value or opposite of the qubit it
                            is being mapped onto.
                        In the above qubit_map, the original qubit 0 is now represented by the opposite
                            value of qubit 2, and the original qubit 1 is now represented by the value of
                            qubit 3.

        Returns:
            List[Tuple[int]] The solution(s) to recursive QAOA as a list of tuples; each tuple is a tuple
                of integer bits.
        """

        n_qubits = count_qubits(change_operator_type(cost_hamiltonian, QubitOperator))

        assert self._n_c < n_qubits

        if not qubit_map:
            qubit_map = []
            for i in range(n_qubits):
                qubit_map.append([i, 1])

        # TODO: is there a better way to allow ansatzes to be modular? Not all
        # ansatzes have a `cost_hamiltonian` attribute.
        # You can't `partial` a constructor, right?
        ansatz = copy(self._ansatz)
        ansatz.cost_hamiltonian = cost_hamiltonian

        # Run QAOA
        estimation_tasks_factory = self._estimation_tasks_factory(
            cost_hamiltonian, ansatz
        )
        cost_function = self._cost_function_factory(
            estimation_tasks_factory,
        )
        opt_results = self._optimizer.minimize(cost_function, self._initial_params)

        # For each term, calculate <psi(beta, gamma) | Z_i Z_j | psi(beta, gamma)>
        # with optimal beta and gamma.
        largest_expval = 0.0

        for term in cost_hamiltonian:
            # If term is a constant term, don't calculate expectation value.
            if () not in term.terms:

                # Calculate expectation value of term
                estimation_tasks_factory_of_term = self._estimation_tasks_factory(
                    term, ansatz
                )
                cost_function_of_term = self._cost_function_factory(
                    estimation_tasks_factory_of_term
                )
                expval_of_term = cost_function_of_term(opt_results.opt_params)

                if np.abs(expval_of_term) > np.abs(largest_expval):
                    largest_expval = expval_of_term
                    term_with_largest_expval = term

        # Loop through all terms again and calculate the mapped result of the term.
        for term in term_with_largest_expval.terms:
            term_with_largest_expval = term
        # term_with_largest_expval is now a subscriptable tuple like ((0, 'Z'), (1, 'Z'))

        qubit_to_get_rid_of: int = term_with_largest_expval[1][0]

        # i is original qubit, qubit_map[i][0] is current qubit evquivalent of original qubit.
        for i in range(len(qubit_map)):
            if qubit_map[i][0] > qubit_to_get_rid_of:
                # map qubit to the qubit 1 below it
                qubit_map[i][0] -= 1
            elif qubit_map[i][0] == qubit_to_get_rid_of:
                # map qubit onto the qubit it's being replaced with
                qubit_map[i][0] = qubit_map[term_with_largest_expval[0][0]][0]
                qubit_map[i][1] *= int(np.sign(largest_expval))

        reduced_cost_hamiltonian = IsingOperator((), 0)

        terms = change_operator_type(cost_hamiltonian, QubitOperator).terms
        for term in terms:
            # term is tuple representing one term of QubitOperator, example ((2, 'Z'), (3, 'Z'))
            if term != term_with_largest_expval:
                coefficient: float = terms[term]
                new_term: Tuple = ()
                for qubit in term:
                    # qubit is a component of qubit operator on 1 qubit ex. (2, 'Z')
                    qubit_indice: int = qubit[0]

                    # Map the new cost hamiltonian onto reduced qubits
                    new_qubit_indice = qubit_map[qubit_indice][0]
                    new_qubit = (new_qubit_indice, "Z")
                    new_term += (new_qubit,)

                    if qubit_indice == qubit_to_get_rid_of:
                        coefficient *= np.sign(largest_expval)

                reduced_cost_hamiltonian += IsingOperator(new_term, coefficient)

        # Check new cost hamiltonian has correct amount of qubits
        assert (
            count_qubits(change_operator_type(reduced_cost_hamiltonian, QubitOperator))
            == count_qubits(change_operator_type(cost_hamiltonian, QubitOperator)) - 1
        )

        # Check qubit map has correct amount of qubits
        assert (
            count_qubits(change_operator_type(reduced_cost_hamiltonian, QubitOperator))
            == max(np.abs(qubit_map).tolist())[0] + 1
        )

        if (
            count_qubits(change_operator_type(reduced_cost_hamiltonian, QubitOperator))
            > self._n_c
        ):
            return self(
                cost_hamiltonian=cost_hamiltonian,
                qubit_map=qubit_map,
            )

        else:
            best_value, reduced_solutions = solve_problem_by_exhaustive_search(
                reduced_cost_hamiltonian
            )
            for solution in reduced_solutions:
                assert len(solution) == count_qubits(
                    change_operator_type(reduced_cost_hamiltonian, QubitOperator)
                )

            # Map the answer of the reduced Hamiltonian back to the original number of qubits.
            solutions: List[Tuple] = []

            for solution in reduced_solutions:
                original_solution: List[int] = []
                for qubit in qubit_map:
                    this_answer = solution[np.abs(qubit[0])]

                    # If negative, flip the qubit.
                    if qubit[1] == -1:
                        if this_answer == 0:
                            this_answer = 1
                        else:
                            this_answer = 0
                    original_solution.append(this_answer)

                solutions.append(tuple(original_solution))

            return solutions
