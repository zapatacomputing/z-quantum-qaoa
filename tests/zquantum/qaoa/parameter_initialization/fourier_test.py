################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
import numpy as np
import pytest
from zquantum.core.cost_function import (
    create_cost_function,
    substitution_based_estimation_tasks_factory,
)
from zquantum.core.estimation import calculate_exact_expectation_values
from zquantum.core.interfaces.ansatz import Ansatz
from zquantum.core.interfaces.cost_function import CostFunction
from zquantum.core.interfaces.functions import (
    CallableWithGradient,
    function_with_gradient,
)
from zquantum.core.interfaces.mock_objects import MockOptimizer, mock_cost_function
from zquantum.core.interfaces.optimizer import optimization_result
from zquantum.core.interfaces.optimizer_test import NESTED_OPTIMIZER_CONTRACTS
from zquantum.core.openfermion import IsingOperator
from zquantum.core.symbolic_simulator import SymbolicSimulator
from zquantum.qaoa.ansatzes import QAOAFarhiAnsatz
from zquantum.qaoa.parameter_initialization import (
    FourierOptimizer,
    convert_u_v_to_gamma_beta,
)
from zquantum.qaoa.parameter_initialization._fourier import _perturb_params_randomly


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
    def _cf_factory(ansatz: Ansatz):
        estimation_tasks_factory = substitution_based_estimation_tasks_factory(
            hamiltonian,
            ansatz,
        )
        return create_cost_function(
            backend=SymbolicSimulator(),
            estimation_tasks_factory=estimation_tasks_factory,
            estimation_method=calculate_exact_expectation_values,  # type: ignore
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
        result = optimization_result(
            opt_params=initial_params,
            opt_value=cost_function(initial_params),  # type: ignore
            nit=1,
            nfev=1,
            history=[],
        )

        # Call the gradient function to make sure it works properly.
        if isinstance(cost_function, CallableWithGradient):
            result.gradient_history = cost_function.gradient(initial_params)

        return result

    inner_optimizer._minimize = custom_minimize
    return inner_optimizer


class TestFourier:
    @pytest.mark.parametrize("contract", NESTED_OPTIMIZER_CONTRACTS)
    def test_if_satisfies_contracts(
        self, contract, ansatz, initial_params, inner_optimizer, cost_function_factory
    ):
        optimizer = FourierOptimizer(
            ansatz=ansatz,
            inner_optimizer=inner_optimizer,
            min_layer=1,
            max_layer=1,
            R=0,
        )

        assert contract(optimizer, cost_function_factory, initial_params)

    @pytest.mark.parametrize(
        ["initial_params", "q"],
        [([[1, 2], [3, 4]], 2), ([1, 2], 2), ([1, 2, 3, 4], None)],
    )
    def test_raises_exception_when_initial_param_size_is_wrong(
        self, ansatz, inner_optimizer, cost_function_factory, initial_params, q
    ):
        initial_params = np.array(initial_params)
        optimizer = FourierOptimizer(
            ansatz=ansatz,
            inner_optimizer=inner_optimizer,
            min_layer=1,
            max_layer=1,
            q=q,
            R=0,
        )
        with pytest.raises(ValueError):
            optimizer.minimize(cost_function_factory, initial_params)

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
            max_layer=1,
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
    def test_get_new_layer_params_returns_correct_params_when_q_is_none(
        self, n_layers_per_iter, ansatz, inner_optimizer
    ):
        # Given
        min_layer = 1
        optimizer = FourierOptimizer(
            ansatz=ansatz,
            inner_optimizer=inner_optimizer,
            min_layer=min_layer,
            max_layer=1,
            n_layers_per_iteration=n_layers_per_iter,
            q=None,
            R=0,
        )
        u_v_params = np.random.uniform(-np.pi, np.pi, min_layer * 2)
        expected_new_params = np.append(u_v_params, np.zeros(n_layers_per_iter * 2))

        # When
        new_params = optimizer._get_u_v_for_next_layer(u_v_params)

        # Then
        assert np.allclose(expected_new_params, new_params)

    def test_records_correct_nit_nfev_and_history_length(
        self, ansatz, inner_optimizer, cost_function_factory
    ):
        min_layer = 1
        max_layer = 2
        optimizer = FourierOptimizer(
            ansatz=ansatz,
            inner_optimizer=inner_optimizer,
            min_layer=min_layer,
            max_layer=max_layer,
            R=0,
        )

        expected_nit = max_layer - min_layer + 1
        # 1 iteration of the inner optimizer each layer. Mock inner optimizer returns
        # `nfev = 1` and `nit = 1` for each optimization run.

        initial_params = np.ones(2)
        opt_result = optimizer.minimize(
            cost_function_factory, initial_params, keep_history=True
        )

        assert (
            opt_result.nit == opt_result.nfev == len(opt_result.history) == expected_nit
        )

    def test_raises_warning_when_gradient_is_not_finite_differences(
        self, ansatz, inner_optimizer, cost_function_factory
    ):
        def my_gradient(params: np.ndarray) -> np.ndarray:
            return np.sqrt(params)

        def cost_function_with_gradients_factory(*args, **kwargs):
            cost_function = cost_function_factory(*args, **kwargs)
            return function_with_gradient(cost_function, my_gradient)

        optimizer = FourierOptimizer(
            ansatz=ansatz,
            inner_optimizer=inner_optimizer,
            min_layer=1,
            max_layer=1,
            R=0,
        )

        initial_params = np.ones(2)

        with pytest.warns(Warning):
            optimizer.minimize(cost_function_with_gradients_factory, initial_params)

    def test_fourier_works_when_cost_function_has_no_gradient(
        self, cost_function_factory, ansatz, inner_optimizer
    ):
        def cost_function_without_gradients_factory(ansatz: Ansatz):
            return cost_function_factory(ansatz).function

        optimizer = FourierOptimizer(
            ansatz=ansatz,
            inner_optimizer=inner_optimizer,
            min_layer=1,
            max_layer=1,
            R=0,
        )
        optimizer.minimize(cost_function_without_gradients_factory, np.ones(2))


class TestPerturbations:
    def test_finds_best_params_from_list(self, ansatz, inner_optimizer):
        params_list = [np.array([i]) for i in [-5, -4, -3, 2, 3, 4, 7, 9]]
        expected_best_params = np.array([2])

        optimizer = FourierOptimizer(
            ansatz=ansatz,
            inner_optimizer=inner_optimizer,
            min_layer=1,
            max_layer=1,
            R=0,
        )
        best_params, best_value, nfev, nit = optimizer._find_best_params_from_list(
            params_list, mock_cost_function
        )

        np.testing.assert_array_equal(best_params, expected_best_params)
        assert best_value == mock_cost_function(expected_best_params)

        # Mock inner optimizer returns `nfev = 1` and `nit = 1` for each cost function's
        # optimization run. This makes sure that `_find_best_params_from_list`
        # increments nit/nfev properly
        assert nfev == nit == len(params_list)

    def test_mean_of_added_perturbations_is_correct(self):
        num_params = 10
        num_repetitions = 1000
        params = np.ones(num_params)

        average_diff = sum(
            [
                (params - _perturb_params_randomly(params)).sum()
                for _ in range(num_repetitions)
            ]
        ) / (num_repetitions * num_params)

        np.testing.assert_allclose(average_diff, 0.0, atol=1e-02)

    @pytest.mark.parametrize("alpha", [0.2, 0.6, 1])
    def test_variance_of_added_perturbations_is_correct(self, alpha):
        num_repetitions = 1000
        params = np.arange(-2, 8)

        sample = np.array(
            [
                (params - _perturb_params_randomly(params, alpha))
                for _ in range(num_repetitions)
            ]
        )
        avg_variance = np.var(sample / alpha, axis=0)

        # According to https://arxiv.org/abs/1812.01041 (pg 17 last paragraph), the
        # variance of the perturbations is given by the input params.
        np.testing.assert_allclose(avg_variance, np.abs(params), rtol=1.5e-1)

    def test_does_not_mutate_parameters(self):
        params = np.ones(4)

        _perturb_params_randomly(params)

        np.testing.assert_array_equal(params, np.ones(4))

    def test_final_opt_result_records_data_from_perturbations(
        self, ansatz, inner_optimizer, cost_function_factory
    ):
        n_perturbations = 2
        min_layer = 1
        max_layer = 2
        optimizer = FourierOptimizer(
            ansatz=ansatz,
            inner_optimizer=inner_optimizer,
            min_layer=min_layer,
            max_layer=max_layer,
            R=n_perturbations,
        )

        expected_nit = 1 + (n_perturbations + 2) * (max_layer - min_layer)
        # expected_nit = n iters on first layer (1) + n iters from perturbations
        # + n iters from 2 params propaged forwards from previous layer. The latter two
        # happen on all layers besides the first layer

        initial_params = np.ones(2)
        opt_result = optimizer.minimize(
            cost_function_factory, initial_params, keep_history=True
        )

        # Mock inner optimizer returns `nfev = 1` and `nit = 1` for each optimization
        # run.
        assert (
            opt_result.nit == opt_result.nfev == len(opt_result.history) == expected_nit
        )
