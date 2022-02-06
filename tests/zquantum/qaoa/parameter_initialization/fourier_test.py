from functools import partial

import numpy as np
import pytest
from openfermion import IsingOperator
from zquantum.core.cost_function import (
    create_cost_function,
    substitution_based_estimation_tasks_factory,
)
from zquantum.core.estimation import (
    allocate_shots_uniformly,
    estimate_expectation_values_by_averaging,
)
from zquantum.core.interfaces.ansatz import Ansatz
from zquantum.core.interfaces.cost_function import CostFunction
from zquantum.core.interfaces.mock_objects import MockOptimizer
from zquantum.core.interfaces.optimizer_test import NESTED_OPTIMIZER_CONTRACTS
from zquantum.core.symbolic_simulator import SymbolicSimulator
from zquantum.qaoa.ansatzes import QAOAFarhiAnsatz
from zquantum.qaoa.parameter_initialization import (
    FourierOptimizer,
    convert_u_v_to_gamma_beta,
)


@pytest.fixture
def initial_params():
    return np.array([1, 2])


@pytest.fixture
def hamiltonian():
    return (
        IsingOperator("Z0 Z1", 5)
        + IsingOperator("Z0 Z3", 2)
        + IsingOperator("Z1 Z2", 0.5)
        + IsingOperator("Z2 Z3", 0.6)
    )


@pytest.fixture
def ansatz(hamiltonian):
    return QAOAFarhiAnsatz(1, hamiltonian)


@pytest.fixture
def cost_function_factory(hamiltonian):
    def _cf_factory(
        ansatz: Ansatz,
    ):
        estimation_preprocessors = [
            partial(allocate_shots_uniformly, number_of_shots=1000)
        ]
        estimation_tasks_factory = substitution_based_estimation_tasks_factory(
            hamiltonian,
            ansatz,
            estimation_preprocessors=estimation_preprocessors,
        )
        return create_cost_function(
            backend=SymbolicSimulator(),
            estimation_tasks_factory=estimation_tasks_factory,
            estimation_method=estimate_expectation_values_by_averaging,
            parameter_preprocessors=None,
        )

    return _cf_factory


@pytest.fixture
def inner_optimizer():
    inner_optimizer = MockOptimizer()

    def custom_minimize(
        cost_function: CostFunction,
        initial_params: np.ndarray,
        keep_history: bool = False,
    ):
        result = MockOptimizer()._minimize(cost_function, initial_params, keep_history)

        # Call the gradient function to make sure it works properly.
        if hasattr(cost_function, "gradient"):
            result.gradient_history = cost_function.gradient(initial_params)

        return result

    inner_optimizer._minimize = custom_minimize
    return inner_optimizer


class TestFouier:
    @pytest.mark.parametrize("contract", NESTED_OPTIMIZER_CONTRACTS)
    def test_if_satisfies_contracts(
        self, contract, ansatz, initial_params, inner_optimizer, cost_function_factory
    ):
        optimizer = FourierOptimizer(
            ansatz=ansatz,
            inner_optimizer=inner_optimizer,
            min_layer=1,
            max_layer=2,
            R=0,
        )

        assert contract(optimizer, cost_function_factory, initial_params)

    @pytest.mark.parametrize("n_layers", [1, 2, 3])
    def test_fourier_returns_correct_param_size(self, n_layers):
        q = np.random.randint(1, 5)
        u_v_params = np.random.uniform(-np.pi, np.pi, q * 2)
        gamma_beta_params = convert_u_v_to_gamma_beta(n_layers, u_v_params)
        assert gamma_beta_params.size == n_layers * 2

    def test_fourier_returns_correct_values(self):
        u_and_v = np.array([1, -0.75, 2, -1.25])
        n_layers = 2
        gammas_and_betas = convert_u_v_to_gamma_beta(n_layers, u_and_v)

        # See equation 8 of https://arxiv.org/abs/1812.01041 for how the expected params
        # are calculated
        expected_gammas_and_betas = np.array(
            [
                np.sin(np.pi / 8) + 2 * np.sin(3 * np.pi / 8),
                -0.75 * np.cos(np.pi / 8) - 1.25 * np.cos(3 * np.pi / 8),
                np.sin(3 * np.pi / 8) + 2 * np.sin(9 * np.pi / 8),
                -0.75 * np.cos(3 * np.pi / 8) - 1.25 * np.cos(9 * np.pi / 8),
            ]
        )
        assert np.allclose(gammas_and_betas, expected_gammas_and_betas)

    @pytest.mark.parametrize("params", [[[0, 1], [1, 0]], [1, 2, 3]])
    def test_fourier_raises_exception_when_param_size_is_wrong(self, params):
        # Given
        params = np.array(params)
        n_layers = 1

        # When/Then
        with pytest.raises(ValueError):
            convert_u_v_to_gamma_beta(n_layers, params)

    @pytest.mark.parametrize("n_layers_per_iter", [1, 3])
    def test_get_new_layer_params_returns_same_params_when_q_is_constant(
        self, n_layers_per_iter, ansatz, inner_optimizer
    ):
        # Given
        q = np.random.randint(1, 5)
        optimizer = FourierOptimizer(
            ansatz=ansatz,
            inner_optimizer=inner_optimizer,
            min_layer=1,
            max_layer=2,
            n_layers_per_iteration=n_layers_per_iter,
            q=q,
            R=0,
        )
        # It doesn't really matter what q is because it's fixed independent of the
        # target gammas/betas size
        u_v_params = np.random.uniform(-np.pi, np.pi, q * 2)

        # When
        new_params = optimizer._get_u_v_for_next_layer(u_v_params)

        # Then
        assert np.allclose(u_v_params, new_params)

    @pytest.mark.parametrize("n_layers_per_iter", [1, 3])
    def test_get_new_layer_params_returns_correct_params_when_q_is_infinity(
        self, n_layers_per_iter, ansatz, inner_optimizer
    ):
        # Given
        min_layer = 1
        optimizer = FourierOptimizer(
            ansatz=ansatz,
            inner_optimizer=inner_optimizer,
            min_layer=min_layer,
            max_layer=2,
            n_layers_per_iteration=n_layers_per_iter,
            q=np.inf,
            R=0,
        )
        u_v_params = np.random.uniform(-np.pi, np.pi, min_layer * 2)
        expected_new_params = np.append(u_v_params, np.zeros(n_layers_per_iter * 2))

        # When
        new_params = optimizer._get_u_v_for_next_layer(u_v_params)

        # Then
        assert np.allclose(expected_new_params, new_params)
