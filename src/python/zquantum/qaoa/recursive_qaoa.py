from zquantum.core.cost_function import AnsatzBasedCostFunction
from zquantum.core.interfaces.ansatz import Ansatz
from zquantum.core.interfaces.backend import QuantumBackend
from zquantum.core.interfaces.optimizer import Optimizer
from zquantum.core.openfermion import change_operator_type
from openfermion import QubitOperator, IsingOperator
from openfermion.utils import count_qubits
from zquantum.core.interfaces.backend import QuantumBackend
import numpy as np
from zquantum.core.measurement import Measurements
from typing import Optional, List, Tuple
from zquantum.qaoa.problems import solve_problem_by_exhaustive_search
from zquantum.qaoa.ansatzes.farhi_ansatz import QAOAFarhiAnsatz
from zquantum.core.bitstring_distribution import BitstringDistribution


class RecursiveQAOA:
    def __init__(
        self,
        cost_hamiltonian: IsingOperator,
    ) -> None:
        """This is an implementation of recursive QAOA (RQAOA) from https://arxiv.org/abs/1910.08980 page 4.

        The main idea is that we call QAOA recursively and reduce the size of the cost hamiltonian by 1
        on each recursion, until we hit a threshold number of qubits `n_c`. Then, we use a brute force method
        to

        Args:
            cost_hamiltonian: Hamiltonian representing the cost function
            n_layers: number of layers in each QAOA ansatz circuit
            n_samples: number of samples to take when calculating expectation values of each QAOA ansatz circuit

        Attributes:
            number_of_qubits: number of qubits used for the initial QAOA circuit
        """
        self._cost_hamiltonian = cost_hamiltonian

    @property
    def number_of_qubits(self):
        """Returns number of qubits used for the QAOA circuit of this recursion."""
        return count_qubits(change_operator_type(self._cost_hamiltonian, QubitOperator))

    def __call__(
        self,
        n_c: int,
        ansatz: Ansatz,
        n_layers: int,
        estimation_method,
        estimation_preprocessors,
        initial_params: np.ndarray,
        optimizer: Optimizer,
        n_samples: int,
        backend: QuantumBackend,
        qubit_map: Optional[List[List[int]]] = None,
    ) -> List[Tuple]:
        """Args:
            n_c: The threshold number of qubits at which recursion stops, as described in the original paper. Cannot be greater than number of qubits.
            qubit_map: A list that maps qubits in reduced Hamiltonian back to original qubits, used for subsequent recursions.
                [(2, -1), (3, 1)]
                first term of tuple is qubit the index of tuple to be mapped onto,
                2nd term is if it will be mapped onto posibive or opposite of the qubit it's being mapped onto.

        Returns:
            List[Tuple[int]] The solution(s) to recursive QAOA as a list of tuples, each tuple is a tuple of bits
        """
        assert n_c < self.number_of_qubits
        if not qubit_map:
            qubit_map = []
            for i in range(self.number_of_qubits):
                qubit_map.append([i, 1])

        # Run QAOA
        # TODO Have ansatz and cost funcs be partials inputs?
        ansatz = QAOAFarhiAnsatz(n_layers, self._cost_hamiltonian)
        cost_function = AnsatzBasedCostFunction(
            self._cost_hamiltonian,
            ansatz,
            backend,
            estimation_method,
            estimation_preprocessors,
        )
        opt_results = optimizer.minimize(cost_function, initial_params)

        # Circuit with optimal parameters.
        circuit = ansatz.get_executable_circuit(opt_results.opt_params)

        # abc = opt_results.opt_params - initial_params
        # return opt_results.opt_value
        # breakpoint()

        # For each term, calculate <psi(beta, gamma) | Z_i Z_j | psi(beta, gamma)>
        # with optimal parameters.
        distribution = backend.get_bitstring_distribution(circuit, n_samples=n_samples)
        largest_expval = 0.0

        for term in self._cost_hamiltonian:
            # Calculate expectation value of term

            # If term is a constant term, don't calculate expectation value.
            if () not in term.terms:
                expval_of_term = _get_expectation_value_of_distribution(
                    distribution, operator=term
                )

                if np.abs(expval_of_term) > np.abs(largest_expval):
                    largest_expval = expval_of_term
                    term_with_largest_expval = term

        # Loop through all terms again and calculate the mapped result of the term.
        for term in term_with_largest_expval.terms:
            term_with_largest_expval = term
        # term_with_largest_expval is now a subscriptable tuple like ((0, 'Z'), (1, 'Z'))

        qubit_to_get_rid_of: int = term_with_largest_expval[1][0]
        # qubit_to_get_rid_of_og = qubit_to_get_rid_of
        # for qubit in range(self.number_of_qubits):
        #     if qubit_map[qubit][0] == qubit_to_get_rid_of:
        #         qubit_to_get_rid_of_og = qubit
        #         break

        breakpoint()
        # i is original qubit, qubit_map[i][0] is current qubit evquivalent of original qubit.
        for i in range(len(qubit_map)):
            if qubit_map[i][0] > qubit_to_get_rid_of:
                # map qubit to the qubit 1 below it
                qubit_map[i][0] -= 1
            elif qubit_map[i][0] == qubit_to_get_rid_of:
                # map qubit onto the qubit it's being replaced with
                qubit_map[i][0] = qubit_map[term_with_largest_expval[0][0]][0]
                qubit_map[i][1] *= int(np.sign(largest_expval))
                breakpoint()

        # After the others are done and `qubit_map[term_with_largest_expval[1][0]][0]` is up to date
        # for i in range(len(qubit_map)):

        # breakpoint()
        # for i in range(qubit_to_get_rid_of_og + 1, len(qubit_map)):
        #     qubit_map[i][0] -= 1
        # qubit_map[qubit_to_get_rid_of] = [
        #     qubit_map[term_with_largest_expval[1][0]][0],
        #     int(np.sign(largest_expval)),
        # ]
        breakpoint()

        new_cost_hamiltonian = IsingOperator((), 0)

        terms = change_operator_type(self._cost_hamiltonian, QubitOperator).terms
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

                new_cost_hamiltonian += IsingOperator(new_term, coefficient)

        # Check new cost hamiltonian has correct amount of qubits
        assert (
            count_qubits(change_operator_type(new_cost_hamiltonian, QubitOperator))
            == count_qubits(change_operator_type(self._cost_hamiltonian, QubitOperator))
            - 1
        )

        # Check qubit map has correct amount of qubits
        assert (
            count_qubits(change_operator_type(new_cost_hamiltonian, QubitOperator))
            == max(np.abs(qubit_map).tolist())[0] + 1
        )

        if (
            count_qubits(change_operator_type(new_cost_hamiltonian, QubitOperator))
            > n_c
        ):
            next_recursion = RecursiveQAOA(new_cost_hamiltonian)
            return next_recursion(
                n_c=n_c,
                ansatz=ansatz,
                n_layers=n_layers,
                estimation_method=estimation_method,
                estimation_preprocessors=estimation_preprocessors,
                initial_params=initial_params,
                optimizer=optimizer,
                n_samples=n_samples,
                backend=backend,
                qubit_map=qubit_map,
            )
        else:
            answers = solve_problem_by_exhaustive_search(new_cost_hamiltonian)
            for answer in answers[1]:
                assert len(answer) == count_qubits(
                    change_operator_type(new_cost_hamiltonian, QubitOperator)
                )

            # Map the answer of the reduced Hamiltonian back to the original number of qubits.
            solutions: List[Tuple] = []

            for answer in answers[1]:
                solution_for_original_qubits: List[int] = []
                for qubit in qubit_map:
                    this_answer = answer[np.abs(qubit[0])]

                    # If negative, flip the qubit.
                    if qubit[1] == -1:
                        if this_answer == 0:
                            this_answer = 1
                        else:
                            this_answer = 0
                    solution_for_original_qubits.append(this_answer)
                # solutions_for_original_qubits is correct given qubit_map and answer

                solutions.append(tuple(solution_for_original_qubits))
            breakpoint()

            return solutions


def _get_expectation_value_of_distribution(
    distribution: BitstringDistribution, operator: IsingOperator
) -> float:
    """Calculates expectation values of a distribution"""
    expectation_values_per_bitstring = {}
    for bitstring in distribution.distribution_dict:

        expected_value = Measurements([bitstring]).get_expectation_values(
            operator,
            use_bessel_correction=False,
        )

        expectation_values_per_bitstring[bitstring] = np.sum(expected_value.values)

    cumulative_value = 0.0
    # Get total expectation value (mean of expectation values of all bitstrings weighted by distribution)
    for bitstring in expectation_values_per_bitstring:
        prob = distribution.distribution_dict[bitstring]
        expectation_value = expectation_values_per_bitstring[bitstring]
        cumulative_value += prob * expectation_value

    return cumulative_value
