import numpy as np
import pytest
from zquantum.qaoa.parameter_initialization import (
    Fourier,
    get_new_layer_params_using_interp,
)


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


class TestFourier:
    @pytest.mark.parametrize("n_layers", [1, 2, 3])
    def test_fourier_returns_correct_param_size(self, n_layers):
        my_fourier_object = Fourier(n_layers=n_layers)
        q = np.random.randint(5)
        u_v_params = np.random.uniform(-np.pi, np.pi, q * 2)
        assert my_fourier_object(u_v_params).size == n_layers * 2

    def test_fourier_returns_correct_values(self):
        my_fourier = Fourier(2)
        gammas_and_betas = my_fourier(np.array([1, -0.75, 2, -1.25]))

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
        my_fourier = Fourier(n_layers=1)

        # When/Then
        with pytest.raises(ValueError):
            my_fourier(params)

    @pytest.mark.parametrize("target_size", [6, 10])
    def test_get_new_layer_params_increments_n_layers_and_returns_correct_params(
        self, target_size
    ):
        # Given
        my_fourier = Fourier(2)
        # It doesn't really matter what q is because it's fixed independent of the
        # target gammas/betas size
        q = np.random.randint(5)
        u_v_params = np.random.uniform(-np.pi, np.pi, q * 2)

        # When
        new_params = my_fourier.get_new_layer_params(target_size, u_v_params)

        # Then
        assert my_fourier.n_layers == target_size // 2
        assert np.allclose(u_v_params, new_params)

    @pytest.mark.parametrize(
        # n_params here represents number of gammas and betas
        "n_params, target_n_params",
        [(4, 2), (3, 2), (4, 0), (2, 3)],
    )
    def test_get_new_layer_params_raises_exception_when_param_sizes_are_wrong(
        self, n_params, target_n_params
    ):
        # Given
        q = 1
        u_v_params = np.random.uniform(-np.pi, np.pi, q * 2)
        parameter_initializer = Fourier(n_layers=n_params // 2).get_new_layer_params

        # When/Then
        with pytest.raises(ValueError):
            parameter_initializer(target_n_params, u_v_params)
