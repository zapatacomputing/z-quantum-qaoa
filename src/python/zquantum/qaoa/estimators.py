import numpy as np
from openfermion import IsingOperator
from typing import List

from zquantum.core.interfaces.backend import QuantumBackend
from zquantum.core.measurement import ExpectationValues, Measurements
from zquantum.core.wip.estimators.estimation_interface import (
    EstimateExpectationValues,
    EstimationTask,
)


class CvarEstimator(EstimateExpectationValues):
    """An estimator for calculating expectation value using CVaR method.
    The main idea is that for diagonal operators the ground state of the Hamiltonian is a base state.
    In particular for the combinatorial optimization problems, we care only about getting a single bitstring representing the solution.
    Therefore, it might be beneficial for the optimization process to discard samples that represent inferior solutions.
    CVaR Estimator takes uses only a top X percentile of the samples for calculating the expectation value, where X = alpha*100.

    Reference: https://arxiv.org/abs/1907.04769
    "Improving Variational Quantum Optimization using CVaR", P. Barkoutsos, G. Nannicini, A. Robert, I. Tavernelli, and S. Woerner
    """

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def __call__(
        self, backend: QuantumBackend, estimation_tasks: List[EstimationTask]
    ) -> List[ExpectationValues]:
        """Given a circuit, backend, and target operators, this method produces expectation values
        using CVaR algorithm.
        TODO
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
        if self.alpha > 1 or self.alpha <= 0:
            raise ValueError("alpha needs to be a value between 0 and 1.")

        circuits, operators, shots_per_circuit = zip(
            *[(e.circuit, e.operator, e.number_of_shots) for e in estimation_tasks]
        )
        distributions_list = [
            backend.get_bitstring_distribution(circuit, n_samples=n_shots)
            for circuit, n_shots in zip(circuits, shots_per_circuit)
        ]

        return [
            ExpectationValues(
                np.array(
                    _calculate_expectation_value_for_distribution(
                        distribution, operator, self.alpha
                    )
                )
            )
            for distribution, operator in zip(distributions_list, operators)
        ]


def _calculate_expectation_value_for_distribution(distribution, operator, alpha):
    expected_values_per_bitstring = {}
    for bitstring in distribution.distribution_dict:
        expected_value = Measurements(bitstring).get_expectation_values(operator)
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
    return final_value
