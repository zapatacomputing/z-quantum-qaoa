import numpy as np
from openfermion import IsingOperator
from typing import Optional
from overrides import overrides

from zquantum.core.circuit import Circuit
from zquantum.core.interfaces.backend import QuantumBackend, QuantumSimulator
from zquantum.core.measurement import ExpectationValues, Measurements
from zquantum.core.interfaces.estimator import Estimator


class CvarEstimator(Estimator):
    """An estimator for calculating expectation value using CVaR method.
    The main idea is that for diagonal operators the ground state of the Hamiltonian is a base state.
    In particular for the combinatorial optimization problems, we care only about getting a single bitstring representing the solution.
    Therefore, it might be beneficial for the optimization process to discard samples that represent inferior solutions.
    CVaR Estimator takes uses only a top X percentile of the samples for calculating the expectation value, where X = alpha*100.

    Reference: https://arxiv.org/abs/1907.04769
    "Improving Variational Quantum Optimization using CVaR", P. Barkoutsos, G. Nannicini, A. Robert, I. Tavernelli, and S. Woerner
    """

    @overrides
    def get_estimated_expectation_values(
        self,
        backend: QuantumBackend,
        circuit: Circuit,
        target_operator: IsingOperator,
        alpha: float,
        n_samples: Optional[int] = None,
    ) -> ExpectationValues:
        """Given a circuit, backend, and target operators, this method produces expectation values
        using CVaR algorithm.

        Args:
            backend (QuantumBackend): the backend that will be used to run the circuit
            circuit (Circuit): the circuit that prepares the state.
            target_operator (SymbolicOperator): Operator to be estimated.
            alpha (float): defines what part of the best measurements should be taken into account in the estimation process.
            n_samples (int): Number of measurements done on the unknown quantum state.

        Raises:
            AttributeError: If backend is not a QuantumSimulator.

        Returns:
            ExpectationValues: expectation values for each term in the target operator.
        """
        if alpha > 1 or alpha <= 0:
            raise ValueError("alpha needs to be a value between 0 and 1.")

        if not isinstance(target_operator, IsingOperator):
            raise TypeError("Operator should be of type IsingOperator.")

        distribution = backend.get_bitstring_distribution(circuit, n_samples=n_samples)
        expected_values_per_bitstring = {}

        for bitstring in distribution.distribution_dict:
            expected_value = Measurements(bitstring).get_expectation_values(
                target_operator
            )
            expected_values_per_bitstring[bitstring] = expected_value.values[0]

        sorted_expected_values_per_bitstring_list = sorted(
            expected_values_per_bitstring.items(), key=lambda item: item[1]
        )

        cumulative_prob = 0.0
        cumulative_value = 0.0

        for bitstring, energy in sorted_expected_values_per_bitstring_list:
            prob = distribution.distribution_dict[bitstring]
            if cumulative_prob + prob < alpha:
                cumulative_prob += prob
                cumulative_value += prob * energy
            else:
                cumulative_value += (alpha - cumulative_prob) * energy
                break
        final_value = cumulative_value / alpha
        return ExpectationValues(final_value)
