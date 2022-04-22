################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
from typing import List, Tuple

import numpy as np


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
