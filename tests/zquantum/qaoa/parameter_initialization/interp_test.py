import numpy as np
import pytest
from zquantum.qaoa.parameter_initialization import get_new_layer_params_using_interp


class TestInterp:
    @pytest.mark.parametrize(
        "n_params, target_n_params", [(4, 2), (3, 2), (4, 0), (2, 3)]
    )
    def test_interp_raises_exception_when_param_sizes_are_wrong(
        self, n_params, target_n_params
    ):
        # Given
        params = np.random.uniform(-np.pi, np.pi, n_params)

        # When/Then
        with pytest.raises(ValueError):
            get_new_layer_params_using_interp(target_n_params, params)

    @pytest.mark.parametrize("n_params, target_n_params", [(2, 4), (2, 10), (8, 12)])
    def test_interp_returns_correct_param_length(self, n_params, target_n_params):
        # Given
        params = np.random.uniform(-np.pi, np.pi, n_params)

        # When
        new_params = get_new_layer_params_using_interp(target_n_params, params)

        # Then
        assert len(new_params) == new_params.shape[0] == target_n_params

    @pytest.mark.parametrize(
        "expected_output, number_of_interpolations",
        [
            (np.array([1, -0.75, 1.5, -1, 2, -1.25]), 1),
            (np.array([1, -0.75, 1.25, -0.875, 1.5, -1, 1.75, -1.125, 2, -1.25]), 3),
        ],
    )
    def test_interp_returns_correct_values(
        self, expected_output, number_of_interpolations
    ):
        # Pen and paper calculation

        # original gammas = [1, 2]
        # expected output gammas (1 interpolation) = [1, 1.5, 2]
        # expected output gammas (3 interpolations) = [1, 1.25, 1.5, 1.75, 2]

        # original betas = [-0.75, -1.25]
        # expected output betas (1 interpolation) = [-0.75, -1, -1.25]
        # expected output betas (3 interpolations) = [-0.75, 0.875, -1, -1.125, -1.25]

        # Given
        params = np.array([1, -0.75, 2, -1.25])

        # When
        target_size = number_of_interpolations * 2 + len(params)
        output = get_new_layer_params_using_interp(target_size, params)

        # Then
        assert np.allclose(output, expected_output)
