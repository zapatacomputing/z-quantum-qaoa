import numpy as np
from openfermion import IsingOperator
from typing import List

from zquantum.core.interfaces.backend import QuantumBackend
from zquantum.core.measurement import ExpectationValues, Measurements
from zquantum.core.bitstring_distribution import BitstringDistribution
from zquantum.core.interfaces.estimation import (
    EstimateExpectationValues,
    EstimationTask,
)


class CvarEstimator(EstimateExpectationValues):
    """An estimator for calculating expectation value using CVaR method.
    The main idea is that for diagonal operators the ground state of the Hamiltonian is a base state.
    In particular for the combinatorial optimization problems, we often care only about getting a single bitstring representing the solution.
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
        using CVaR method.

        Args:
            backend: the backend that will be used to run the circuits
            estimation_tasks: the estimation tasks defining the problem. Each task consist of target operator, circuit and number of shots.
            alpha: defines what part of the measurements should be taken into account in the estimation process.
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
                    [
                        _calculate_expectation_value_for_distribution(
                            distribution, operator, self.alpha
                        )
                    ]
                )
            )
            for distribution, operator in zip(distributions_list, operators)
        ]


def _calculate_expectation_value_for_distribution(
    distribution: BitstringDistribution, operator: IsingOperator, alpha: float
) -> float:
    # Calculates expectation value per bitstring
    expectation_values_per_bitstring = {}
    for bitstring in distribution.distribution_dict:
        expected_value = Measurements([bitstring]).get_expectation_values(
            operator, use_bessel_correction=False
        )
        expectation_values_per_bitstring[bitstring] = np.sum(expected_value.values)

    # Sorts expectation values by values.
    sorted_expectation_values_per_bitstring_list = sorted(
        expectation_values_per_bitstring.items(), key=lambda item: item[1]
    )

    cumulative_prob = 0.0
    cumulative_value = 0.0
    # Sums expectation values for each bitstring, starting from the one with the smallest one.
    # When the cumulative probability associated with these bitstrings is higher than alpha,
    # it stops and effectively discards all the remaining values.
    for bitstring, expectation_value in sorted_expectation_values_per_bitstring_list:
        prob = distribution.distribution_dict[bitstring]
        if cumulative_prob + prob < alpha:
            cumulative_prob += prob
            cumulative_value += prob * expectation_value
        else:
            cumulative_value += (alpha - cumulative_prob) * expectation_value
            break
    final_value = cumulative_value / alpha
    return final_value
