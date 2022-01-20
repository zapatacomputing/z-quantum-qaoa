from typing import List, Tuple

import numpy as np
from zquantum.core.interfaces.cost_function import ParameterPreprocessor


def _get_param(params_vector: List[float], layer_number):
    """0 <= layer_number <= p + 1"""
    if layer_number == 0 or layer_number == len(params_vector) + 1:
        return 0
    else:
        return params_vector[layer_number - 1]


def _perform_one_interpolation(
    gammas: List[float], betas: List[float], p: int
) -> Tuple[List[float], List[float]]:
    """Following eq (B1) in the original paper, p is the length of input params."""
    assert len(gammas) == len(betas) == p
    new_gammas = []
    new_betas = []

    for i in range(1, p + 2):
        # See eq (B1) in the original paper.
        bef = i - 1
        new_gamma_i = (
            bef * _get_param(gammas, bef) / p + (p - bef) * _get_param(gammas, i) / p
        )
        new_beta_i = (
            bef * _get_param(betas, bef) / p + (p - bef) * _get_param(betas, i) / p
        )

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
        new_gammas, new_betas = _perform_one_interpolation(
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


def get_new_layer_params_using_fourier(
    number_of_new_params: int, old_params: np.ndarray
) -> np.ndarray:
    return old_params


class Fourier(ParameterPreprocessor):
    """The FOURIER method for initializing QAOA parameters, from
    https://arxiv.org/abs/1812.01041. See Appendix B.
    To be used with LayerwiseAnsatzOptimizerWithFactories. Example use case:
        cost_hamiltonian = ...
        n_layers = ...
        ansatz = QAOAFarhiAnsatz(n_layers, cost_hamiltonian)
        estimation_tasks_factory_generator = partial(
            substitution_based_estimation_tasks_factory,
            target_operator=cost_hamiltonian,
        )
        cost_function_factory = partial(
            create_cost_function,
            backend=...,
            estimation_method=...,
        )
        # The number of initial parameters you use dictate the length of Fourier
        # params u and v
        initial_params = np.array([0.42, 2.2])
        inner_optimizer = ScipyOptimizer(method="L-BFGS-B")
        fourier_optimizer = LayerwiseAnsatzOptimizerWithFactories(
            ansatz,
            inner_optimizer,
            estimation_tasks_factory_generator,
            cost_function_factory,
            min_layer=...,
            max_layer=...,
            parameters_initializer=get_new_layer_params_using_fourier,
        )
        opt_result = fourier_optimizer.minimize(
            initial_params, parameter_initializer=Fourier(n_layers=...)
        )
        opt_params = opt_result.opt_params
    """

    def __init__(self, n_layers: int) -> None:
        self.n_layers = n_layers

    def __call__(self, parameters: np.ndarray) -> np.ndarray:
        assert self.n_layers > 0
        assert len(parameters) % 2 == 0

        # input parameters are 2q u and v parameters.
        q = len(parameters) // 2
        gammas_and_betas = []

        # Calculate gamma of each layer given u and
        # Calculate beta of each layer given v,
        # see eq (8) of original paper.
        for i in range(self.n_layers):
            gamma = 0
            beta = 0
            for n in range(0, q * 2, 2):
                k = n / 2
                gamma += parameters[n] * np.sin(
                    (k - 0.5) * (i - 0.5) * np.pi / self.n_layers
                )
                beta += parameters[n + 1] * np.cos(
                    (k - 0.5) * (i - 0.5) * np.pi / self.n_layers
                )
            gammas_and_betas.append(gamma)
            gammas_and_betas.append(beta)

        # output parameters are 2p parameters.
        assert len(gammas_and_betas) == self.n_layers * 2
        return np.array(gammas_and_betas)
