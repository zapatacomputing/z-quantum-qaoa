import pytest
from pyquil import Program
from pyquil.gates import X, H
from openfermion import QubitOperator, qubit_operator_sparse, IsingOperator
import numpy as np

from zquantum.core.interfaces.estimator_test import EstimatorTests
from zquantum.core.interfaces.mock_objects import (
    MockQuantumBackend,
    MockQuantumSimulator,
)
from zquantum.core.estimator_test import TestBasicEstimator
from zquantum.core.circuit import Circuit
from zquantum.core.measurement import Measurements, ExpectationValues
from .estimators import CvarEstimator


class TestCvarEstimator(EstimatorTests):
    @pytest.fixture()
    def estimator(self, request):
        return CvarEstimator()

    @pytest.fixture(params=[1.0, 0.8, 0.5, 0.2])
    def alpha(self, request):
        return request.param

    @pytest.fixture()
    def target_operator(self, request):
        return IsingOperator("Z0")

    @pytest.fixture()
    def circuit(self, request):
        return Circuit(Program(X(0)))

    @pytest.fixture()
    def backend(self, request):
        backend = MockQuantumBackend()

        def custom_run_circuit_and_measure(circuit, n_samples):
            bitstrings = [("0"), ("1")]
            return Measurements(bitstrings)

        backend.run_circuit_and_measure = custom_run_circuit_and_measure
        return backend

    @pytest.fixture()
    def n_samples(self, request):
        return 10

    def test_get_estimated_expectation_values_returns_expectation_values(
        self, estimator, backend, circuit, target_operator, alpha
    ):
        value = estimator.get_estimated_expectation_values(
            backend=backend,
            circuit=circuit,
            target_operator=target_operator,
            alpha=alpha,
        )
        # Then
        assert type(value) is ExpectationValues

    def test_raises_exception_if_operator_is_not_ising(
        self, estimator, backend, circuit, alpha
    ):
        # Given
        target_operator = QubitOperator("X0")
        with pytest.raises(TypeError):
            value = estimator.get_estimated_expectation_values(
                backend=backend,
                circuit=circuit,
                target_operator=target_operator,
                alpha=alpha,
            ).values

    def test_cvar_estimator_raises_exception_if_alpha_less_than_0(
        self, estimator, backend, circuit, target_operator
    ):
        alpha = -1
        with pytest.raises(ValueError):
            value = estimator.get_estimated_expectation_values(
                backend=backend,
                circuit=circuit,
                target_operator=target_operator,
                alpha=alpha,
            ).values

    def test_cvar_estimator_raises_exception_if_alpha_greater_than_1(
        self, estimator, backend, circuit, target_operator
    ):
        alpha = 2
        with pytest.raises(ValueError):
            value = estimator.get_estimated_expectation_values(
                backend=backend,
                circuit=circuit,
                target_operator=target_operator,
                alpha=alpha,
            ).values

    def test_cvar_estimator_returns_correct_values(
        self, estimator, backend, target_operator, alpha
    ):
        # Given
        circuit = Circuit(Program(H(0)))
        if alpha <= 0.5:
            target_value = -1
        else:
            target_value = (-1 * 0.5 + 1 * (alpha - 0.5)) / alpha

        # When
        value = estimator.get_estimated_expectation_values(
            backend=backend,
            circuit=circuit,
            target_operator=target_operator,
            alpha=alpha,
        ).values

        # Then
        assert value == pytest.approx(target_value)
