from typing import List, Tuple

import numpy as np
from zquantum.core.interfaces.cost_function import ParameterPreprocessor


def _get_param(params_vector: List[float], layer_number):
    """0 <= layer_number <= p + 1"""
    if layer_number == 0 or layer_number == len(params_vector) + 1:
        return 0
    else:
        return params_vector[layer_number - 1]


def _perform_single_interpolation(
    gammas: List[float], betas: List[float], p: int
) -> Tuple[List[float], List[float]]:
    """Following eq (B1) in the original paper, p is the length of input params."""
    assert len(gammas) == len(betas) == p
    new_gammas = []
    new_betas = []

    for i in range(1, p + 2):
        # See eq (B1) in the original paper.
        new_gamma_i = (i - 1) * _get_param(gammas, i - 1) / p + (
            p - (i - 1)
        ) * _get_param(gammas, i) / p
        new_beta_i = (i - 1) * _get_param(betas, i - 1) / p + (
            p - (i - 1)
        ) * _get_param(betas, i) / p

        new_gammas.append(new_gamma_i)
        new_betas.append(new_beta_i)

    return (new_gammas, new_betas)


def get_new_layer_params_using_interp(
    target_size: int, old_params: np.ndarray
) -> np.ndarray:
    """The INTERP method for initializing QAOA parameters, from
    https://arxiv.org/abs/1812.01041. See Appendix B.
    To be used with LayerwiseAnsatzOptimizer as the `parameters_initializer`.

    Args:
        target_size: number of returned parameters
        old_params: params that we want to extend
    """

    if not target_size > len(old_params):
        raise ValueError("Target size must be larger than old params.")
    if not len(old_params) % 2 == target_size % 2 == 0:
        raise ValueError("Size of both old and target parameters must be even.")

    # p is the number of layers of the old circuit
    p = len(old_params) // 2
    number_added_layers = target_size // 2 - p

    old_gammas = []
    old_betas = []
    # Gamma comes before beta of each layer
    for i, param in enumerate(old_params):
        if i % 2 == 0:
            old_gammas.append(param)
        else:
            old_betas.append(param)

    for index in range(number_added_layers):
        new_gammas, new_betas = _perform_single_interpolation(
            old_gammas, old_betas, p + index
        )
        old_gammas = new_gammas
        old_betas = new_betas

    # Add betas and gammas to new params
    new_params = []
    for i in range(target_size):
        if i % 2 == 0:
            new_params.append(new_gammas[i // 2])
        else:
            new_params.append(new_betas[i // 2])

    assert len(new_params) == target_size
    return np.array(new_params)


class Fourier(ParameterPreprocessor):
    """The FOURIER method for initializing QAOA parameters, from
    https://arxiv.org/abs/1812.01041.

    How it works: Fourier uses parameters `u` and `v` to create `gamma` and `beta`,
    which are used to evaluate QAOA. The optimizer is given `u` and `v` instead of
    `gamma` and `beta`. Instead of being length `(2 * n_layers)`, `u` and `v` are of
    length `(2 * q)` where `q` is a hyperparameter. `q` is fixed at the same number when
    n_layers increases. Once `u` and `v` have been sufficiently optimized for the
    current layer, they are used to generate the parameters of the next layer, which
    should provide a good initial parameters for larger layers of QAOA.

    For more detail on how Fourier works, see Appendix B of the original paper.

    The `__call__` method takes `u` and `v` parameters as input and outputs `gamma` and
    `beta`. The size of the output `gamma` and `beta` are dependent on self.n_layers.
    The `get_new_layer_params` method increments the `n_layers` property of the
    `Fourier` object such that the output params of `__call__` matches `target_size`.

    To be used with LayerwiseAnsatzOptimizer. For Fourier to work properly, please do
    the following:
        - Your instance of Fourier should be a parameter processor of cost function
        - The get_new_layer_params method should be the `parameters_initializer`
            argument of LayerwiseAnsatzOptimizer
        - The Fourier instance used as the parameter processor and the parameter
            initializer should be the same, or else the number of layers won't increment
            properly.
        - The initial parameters you give to an optimizer will not be of size
            (2 * n_layers), but of length 2q. q is set depending on the size of initial
            parameters.

    If that was confusing, here's an example use case:
        cost_hamiltonian = ...
        initial_n_layers = ...
        ansatz = QAOAFarhiAnsatz(initial_n_layers, cost_hamiltonian)
        my_fourier_object = Fourier(n_layers=initial_n_layers)

        def cost_function_factory(ansatz: Ansatz):
            estimation_tasks_factory = substitution_based_estimation_tasks_factory(
                cost_hamiltonian,
                ansatz,
            )
            function_with_gradient = create_cost_function(
                backend,
                estimation_tasks_factory,
                calculate_exact_expectation_values,
                parameter_preprocessors=[my_fourier_object]
            )

        # The number of initial parameters you use dictate the length of Fourier
        # params u and v
        # Here, q = 1 because the length of initial_params is 2
        initial_params = np.array([0.42, 2.2])

        inner_optimizer = ScipyOptimizer(method="L-BFGS-B")
        fourier_optimizer = LayerwiseAnsatzOptimizer(
            ansatz,
            inner_optimizer,
            min_layer=initial_n_layers,
            max_layer=...,
            parameters_initializer=my_fourier_object.get_new_layer_params
        )
        opt_result = fourier_optimizer.minimize(
            cost_function_factory, initial_params,
        )
        opt_params = opt_result.opt_params
    """

    def __init__(self, n_layers: int) -> None:
        self.n_layers = n_layers

    def __call__(self, parameters: np.ndarray) -> np.ndarray:
        assert self.n_layers > 0
        if not len(parameters.shape) == 1:
            raise ValueError("Parameters must be a 1d array.")
        if not parameters.size % 2 == 0:
            raise ValueError("Size of parameters must be even.")

        # input parameters are u and v parameters of size 2q.
        q = parameters.size // 2
        gammas_and_betas = []

        u = parameters.reshape(-1, 2).T[0]
        v = parameters.reshape(-1, 2).T[1]

        # Calculate gamma of each layer given `u` and
        # Calculate beta of each layer given `v`,
        # see eq (8) of original paper.
        for i in range(self.n_layers):
            gamma = u.dot(
                np.sin(np.pi / self.n_layers * (np.arange(q) + 0.5) * (i + 0.5))
            )
            beta = v.dot(
                np.cos(np.pi / self.n_layers * (np.arange(q) + 0.5) * (i + 0.5))
            )
            gammas_and_betas.append(gamma)
            gammas_and_betas.append(beta)

        # output parameters are of size (2 * n_layers).
        assert len(gammas_and_betas) == self.n_layers * 2
        return np.array(gammas_and_betas)

    def get_new_layer_params(self, target_size: int, params: np.ndarray) -> np.ndarray:
        # The input `params` are actually of length 2q rather than (2 * n_layers).
        # Returns the same params as input params because q doesn't change.
        if not len(params) % 2 == target_size % 2 == 0:
            raise ValueError("Size of both old and target parameters must be even.")
        current_length_of_gammas_and_betas = self.n_layers * 2
        if not current_length_of_gammas_and_betas < target_size:
            raise ValueError("Target size must be larger than old params.")
        number_of_new_params = target_size - current_length_of_gammas_and_betas
        self.n_layers += number_of_new_params // 2
        return params
