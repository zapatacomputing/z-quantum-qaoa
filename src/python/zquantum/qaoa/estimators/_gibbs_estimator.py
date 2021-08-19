from typing import Dict, List, Optional

import numpy as np
from openfermion import IsingOperator
from pyquil.wavefunction import Wavefunction
from zquantum.core.bitstring_distribution import BitstringDistribution
from zquantum.core.interfaces.backend import QuantumBackend
from zquantum.core.interfaces.estimation import (
    EstimateExpectationValues,
    EstimationTask,
)
from zquantum.core.measurement import ExpectationValues, Measurements
from zquantum.core.utils import dec2bin

from ._utils import (
    _calculate_expectation_value_for_distribution,
    _calculate_expectation_value_for_wavefunction,
)


class GibbsObjectiveEstimator(EstimateExpectationValues):
    """An estimator for calculating expectation value using the Gibbs objective function method.
    The main idea is that we exponentiate the negative expectation value of each sample, which amplifies bitstrings with
    low energies while reducing the role that high energy bitstrings play in determining the cost.

    Reference: https://arxiv.org/abs/1909.07621 Section III
    "Quantum Optimization with a Novel Gibbs Objective Function and Ansatz Architecture Search", L. Li, M. Fan, M. Coram, P. Riley, and S. Leichenauer
    """

    def __init__(
        self, alpha: float, use_exact_expectation_values: Optional[bool] = False
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.use_exact_expectation_values = use_exact_expectation_values

    def __call__(
        self, backend: QuantumBackend, estimation_tasks: List[EstimationTask]
    ) -> List[ExpectationValues]:
        """Given a circuit, backend, and target operators, this method produces expectation values
        using Gibbs objective function.

        Args:
            backend: the backend that will be used to run the circuits
            estimation_tasks: the estimation tasks defining the problem. Each task consist of target operator, circuit and number of shots.
            alpha: defines to exponent coefficient, `exp(-alpha * expectation_value)`. See equation 2 in the original paper.
        """
        if self.alpha <= 0:
            raise ValueError("alpha needs to be a value greater than 0.")

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
                                distribution,
                                operator,
                                self.alpha,
                                _sum_expectation_values,
                            )
                        ]
                    )
                )
                for distribution, operator in zip(distributions_list, operators)
            ]

        else:
            wavefunctions_list = [
                backend.get_wavefunction(circuit) for circuit in circuits
            ]

            return [
                ExpectationValues(
                    np.array(
                        [
                            _calculate_expectation_value_for_wavefunction(
                                distribution,
                                operator,
                                self.alpha,
                                _sum_expectation_values,
                            )
                        ]
                    )
                )
                for distribution, operator in zip(wavefunctions_list, operators)
            ]


def _sum_expectation_values(
    expectation_values_per_bitstring: Dict[str, float],
    probability_per_bitstring: Dict[str, float],
    alpha: float,
) -> float:
    cumulative_value = 0.0
    # Get total expectation value (mean of expectation values of all bitstrings weighted by distribution)
    for bitstring in expectation_values_per_bitstring:
        prob = probability_per_bitstring[bitstring]

        # For the i-th sampled bitstring, compute exp(-alpha E_i) See equation 2 in the original paper.
        expectation_value = np.exp(-alpha * expectation_values_per_bitstring[bitstring])
        cumulative_value += prob * expectation_value

    final_value = -np.log(cumulative_value)

    return final_value
