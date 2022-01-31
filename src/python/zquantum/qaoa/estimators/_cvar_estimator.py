from typing import Dict, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
from openfermion import IsingOperator
from zquantum.core.bitstring_distribution import BitstringDistribution
from zquantum.core.interfaces.backend import QuantumBackend, QuantumSimulator
from zquantum.core.interfaces.estimation import (
    EstimateExpectationValues,
    EstimationTask,
)
from zquantum.core.measurement import ExpectationValues, check_parity_of_vector
from zquantum.core.utils import dec2bin
from zquantum.core.wavefunction import Wavefunction

Bitstring = TypeVar("Bitstring", str, Sequence[int], Tuple[int, ...])
PROBABILITY_CUTOFF = 1e-8


class CvarEstimator(EstimateExpectationValues):
    def __init__(
        self, alpha: float, use_exact_expectation_values: Optional[bool] = False
    ) -> None:
        """An estimator for calculating expectation value using CVaR method.

        The main idea is that for diagonal operators the ground state of the
        Hamiltonian is a base state. In particular for the combinatorial optimization
        problems, we often care only about getting a single bitstring representing the
        solution. Therefore, it might be beneficial for the optimization process to
        discard samples that represent inferior solutions. CVaR Estimator takes uses
        only a top X percentile of the samples for calculating the expectation value,
        where X = alpha*100.

        Reference: https://arxiv.org/abs/1907.04769
        "Improving Variational Quantum Optimization using CVaR",
        P. Barkoutsos, G. Nannicini, A. Robert, I. Tavernelli, and S. Woerner

        Args:
            alpha: defines what part of the measurements should be taken into account
                in the estimation process.
            use_exact_expectation_values: whether to calculate expectation values by
                using exact wavefunctions or by taking samples. (If True, the number of
                 shots in each estimation task will be disregarded.)
        """
        super().__init__()
        self.alpha = alpha
        self.use_exact_expectation_values = use_exact_expectation_values

    def __call__(
        self, backend: QuantumBackend, estimation_tasks: List[EstimationTask]
    ) -> List[ExpectationValues]:
        """Compute expectation value using CVaR method.

        Args:
            backend: the backend that will be used to run the circuits
            estimation_tasks: the estimation tasks defining the problem. Each task
                consist of target operator, circuit and number of shots.
        """
        if self.alpha > 1 or self.alpha <= 0:
            raise ValueError("alpha needs to be a value between 0 and 1.")

        circuits, operators, shots_per_circuit = zip(
            *[(e.circuit, e.operator, e.number_of_shots) for e in estimation_tasks]
        )

        if not self.use_exact_expectation_values:
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

        else:
            if issubclass(type(backend), QuantumSimulator):
                wavefunctions_list = [
                    backend.get_wavefunction(circuit) for circuit in circuits
                ]

                return [
                    ExpectationValues(
                        np.array(
                            [
                                _calculate_expectation_value_for_wavefunction(
                                    distribution, operator, self.alpha
                                )
                            ]
                        )
                    )
                    for distribution, operator in zip(wavefunctions_list, operators)
                ]
            else:
                raise TypeError(
                    "In order to use exact expectation values "
                    "you need to use QuantumSimulator."
                )


def _calculate_expectation_value_for_distribution(
    distribution: BitstringDistribution, operator: IsingOperator, alpha: float
) -> float:
    # Calculates expectation value per bitstring
    expectation_values = _calculate_expectation_values(
        np.array([*distribution.distribution_dict.keys()]), operator
    )

    # Map expectation values back to original bitstrings
    expectation_values_dict = {
        bitstring: float(expectation_values[i])
        for i, bitstring in enumerate(distribution.distribution_dict.keys())
    }
    return _sum_expectation_values(
        expectation_values_dict, distribution.distribution_dict, alpha
    )


def _calculate_expectation_value_for_wavefunction(
    wavefunction: Wavefunction, operator: IsingOperator, alpha: float
) -> float:
    n_qubits = wavefunction.amplitudes.shape[0].bit_length() - 1

    # Compute the probability p(x) for each n-bitstring x from the wavefunction,
    # p(x) = |amplitude of x| ^ 2.
    probability_per_bitstring = np.abs(wavefunction.amplitudes) ** 2

    # Get the bitstrings with non-zero elements in the wavefunction and calculate
    # their expectation values
    integer_bitstrings = (probability_per_bitstring > PROBABILITY_CUTOFF).nonzero()[0]
    bitstrings_array = np.array([dec2bin(n, n_qubits) for n in integer_bitstrings])
    expectation_values = _calculate_expectation_values(bitstrings_array, operator)
    expectation_values_dict = {
        integer_bitstrings[i]: v for i, v in enumerate(expectation_values)
    }
    probability_per_bitstring_dict = {
        integer_bitstrings[i]: v for i, v in enumerate(probability_per_bitstring)
    }

    return _sum_expectation_values(
        expectation_values_dict, probability_per_bitstring_dict, alpha
    )


def _sum_expectation_values(
    expectation_values_per_bitstring: Dict[Bitstring, float],
    probability_per_bitstring: Dict[Bitstring, float],
    alpha: float,
) -> float:
    """Compute cumulative sum of expectation values until probability exceeds alpha.

    Args:
        expectation_values_per_bitstring: dictionary of bitstrings and their
            corresponding expectation values.
        probability_per_bitstring: dictionary of bitstrings and their corresponding
            expectation probabilities.
        alpha: see description in the `__call__()` method.
    """
    # Sorts expectation values by values.
    sorted_expectation_values_per_bitstring_list = sorted(
        expectation_values_per_bitstring.items(), key=lambda item: item[1]
    )

    cumulative_prob = 0.0
    cumulative_value = 0.0
    # Sums expectation values for each bitstring, starting from the one with the
    # smallest one. When the cumulative probability associated with these bitstrings
    # is higher than alpha, it stops and effectively discards all the remaining values.
    for bitstring, expectation_value in sorted_expectation_values_per_bitstring_list:
        prob = probability_per_bitstring[bitstring]
        if cumulative_prob + prob < alpha:
            cumulative_prob += prob
            cumulative_value += prob * expectation_value
        else:
            cumulative_value += (alpha - cumulative_prob) * expectation_value
            break
    final_value = cumulative_value / alpha
    return final_value


def _calculate_expectation_values(
    bitstrings: np.ndarray, operator: IsingOperator
) -> np.ndarray:
    """Calculates expectation values for each bitstring in the given array"""

    if not isinstance(operator, IsingOperator):
        raise TypeError("Input operator not openfermion.IsingOperator")

    expectation_values_list = [
        coefficient
        * (check_parity_of_vector(bitstrings, [op[0] for op in term]) * 2 - 1)
        for term, coefficient in operator.terms.items()
    ]
    return np.array(expectation_values_list).sum(axis=0)
