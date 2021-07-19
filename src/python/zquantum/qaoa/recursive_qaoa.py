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
        """Returns number of qubits used for the initial QAOA circuit."""
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
        qubit_map: Optional[List[int]] = [],
        # TODO: better way of having these params?
    ) -> IsingOperator:
        """Args:
            n_c: The threshold number of qubits at which recursion stops, as described in the original paper. Cannot be greater than number of qubits.
            qubit_map: A list that maps qubits in reduced Hamiltonian back to original qubits, used for subsequent recursions.

        Returns:
            The solution to recursive QAOA as a bitstring. (tuple?)
        """
        assert n_c < self.number_of_qubits
        if not qubit_map:
            for i in range(self.number_of_qubits):
                qubit_map.append(i)

        # Run QAOA
        ansatz = ansatz(n_layers, self._cost_hamiltonian)
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

        # For each term, calculate <psi(beta, gamma) | Z_i Z_j | psi(beta, gamma)>
        # with optimal parameters.
        distribution = backend.get_bitstring_distribution(circuit, n_samples=n_samples)
        largest_expval = 0.0

        for term in self._cost_hamiltonian:
            # Calculate expectation value of term
            # TODO: Allow usuage of different objective functions (CVaR, Gibbs)

            # If term is a constant term, don't calculate expectation value.
            if () not in term.terms:
                expectation_values_per_bitstring = {}
                for bitstring in distribution.distribution_dict:

                    expected_value = Measurements([bitstring]).get_expectation_values(
                        change_operator_type(term, IsingOperator),
                        use_bessel_correction=False,  # TODO is this the correct use of term. term is the operator
                    )

                    expectation_values_per_bitstring[bitstring] = np.sum(
                        expected_value.values
                    )

                cumulative_value = 0.0
                # Get total expectation value (mean of expectation values of all bitstrings weighted by distribution)
                for bitstring in expectation_values_per_bitstring:
                    prob = distribution.distribution_dict[bitstring]
                    expectation_value = expectation_values_per_bitstring[bitstring]
                    cumulative_value += prob * expectation_value

                if np.abs(cumulative_value) > np.abs(largest_expval):
                    largest_expval = cumulative_value
                    term_with_largest_expval = term

        # Loop through all terms again and calculate the mapped result of the term.
        for term in term_with_largest_expval.terms:
            term_with_largest_expval = term
        # term_with_largest_expval is now a subscriptable tuple like ((0, 'Z'), (1, 'Z'))

        import pdb

        pdb.set_trace()

        qubit_to_get_rid_of: int = term_with_largest_expval[0][0]
        for i in range(qubit_to_get_rid_of + 1, len(qubit_map)):
            qubit_map[i] -= 1
        qubit_map[qubit_to_get_rid_of] = qubit_map[
            term_with_largest_expval[1][0]
        ] * int(np.sign(largest_expval))

        import pdb

        pdb.set_trace()

        new_cost_hamiltonian = IsingOperator((), 0)

        terms = change_operator_type(self._cost_hamiltonian, QubitOperator).terms
        for term in terms:
            # term is tuple representing one term of QubitOperator, example ((2, 'Z'), (3, 'Z'))
            if term != term_with_largest_expval:
                coefficient: float = terms[term]
                new_term: Tuple = ()
                for qubit in term:
                    # qubit is a component of qubit operator on 1 qubit ex. (2, 'Z')
                    new_qubit_indice: int = qubit[0]
                    if new_qubit_indice == qubit_to_get_rid_of:
                        new_qubit_indice = term_with_largest_expval[1][0]

                    # Map the qubit onto ...
                    new_qubit_indice = qubit_map[new_qubit_indice]
                    # if new_qubit_indice < 0:
                    #     coefficient *= -1
                    new_qubit_indice = int(np.abs(new_qubit_indice))
                    # bc np.abs returns type 'numpy.int32'
                    new_qubit = (new_qubit_indice, "Z")
                    new_term += (new_qubit,)

                new_cost_hamiltonian += IsingOperator(
                    new_term, np.sign(largest_expval) * coefficient
                )

        # Now I have to make new cost hamiltonian fucking reduced to basic terms I guess.

        import pdb

        pdb.set_trace()
        assert (
            count_qubits(change_operator_type(new_cost_hamiltonian, QubitOperator))
            == max(qubit_map) + 1
        )

        assert (
            count_qubits(change_operator_type(new_cost_hamiltonian, QubitOperator))
            == count_qubits(change_operator_type(self._cost_hamiltonian, QubitOperator))
            - 1
        )

        if (
            count_qubits(change_operator_type(new_cost_hamiltonian, QubitOperator))
            > n_c
        ):
            next_recursion = RecursiveQAOA(new_cost_hamiltonian)
            return next_recursion(
                n_c,
                cost_function,
                initial_params,
                optimizer,
                n_samples,
                backend,
                qubit_map,
            )
        else:
            return new_cost_hamiltonian
            # TODO: brute force the answer and map answer onto original qubits.
            # TODO: make sure it works when highest expval is negative.
