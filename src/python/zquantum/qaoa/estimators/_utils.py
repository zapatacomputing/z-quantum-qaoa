from typing import Callable

import numpy as np
from openfermion import IsingOperator
from pyquil.wavefunction import Wavefunction
from zquantum.core.bitstring_distribution import BitstringDistribution
from zquantum.core.measurement import Measurements
from zquantum.core.utils import dec2bin


def _calculate_expectation_value_of_bitstring(
    bitstring: str, operator: IsingOperator
) -> float:
    """Calculate expectation value for a bitstring based on an operator."""
    expected_value = Measurements([bitstring]).get_expectation_values(
        operator, use_bessel_correction=False
    )
    return np.sum(expected_value.values)


def _calculate_expectation_value_for_distribution(
    distribution: BitstringDistribution,
    operator: IsingOperator,
    alpha: float,
    _sum_expectation_values: Callable,
) -> float:
    # Calculates expectation value per bitstring
    expectation_values_per_bitstring = {}
    for bitstring in distribution.distribution_dict:
        expected_value = _calculate_expectation_value_of_bitstring(bitstring, operator)
        expectation_values_per_bitstring[bitstring] = expected_value

    return _sum_expectation_values(
        expectation_values_per_bitstring, distribution.distribution_dict, alpha
    )


def _calculate_expectation_value_for_wavefunction(
    wavefunction: Wavefunction,
    operator: IsingOperator,
    alpha: float,
    _sum_expectation_values: Callable,
) -> float:
    expectation_values_per_bitstring = {}
    probability_per_bitstring = {}

    n_qubits = wavefunction.amplitudes.shape[0].bit_length() - 1

    for decimal_bitstring in range(2 ** n_qubits):
        # `decimal_bitstring` is the bitstring converted to decimal.

        # Convert decimal bitstring into bitstring
        bitstring = "".join([str(int) for int in dec2bin(decimal_bitstring, n_qubits)])

        # Calculate expectation values for each bitstring.
        expected_value = _calculate_expectation_value_of_bitstring(bitstring, operator)
        expectation_values_per_bitstring[bitstring] = expected_value

        # Compute the probability p(x) for each n-bitstring x from the wavefunction,
        # p(x) = |amplitude of x| ^ 2.
        probability = np.abs(wavefunction.amplitudes[decimal_bitstring]) ** 2
        probability_per_bitstring[bitstring] = float(probability)

    return _sum_expectation_values(
        expectation_values_per_bitstring, probability_per_bitstring, alpha
    )
