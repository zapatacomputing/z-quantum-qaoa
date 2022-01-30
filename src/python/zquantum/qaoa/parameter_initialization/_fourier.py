import copy
from collections import defaultdict
from typing import Callable, Dict, List, Union

import numpy as np
from scipy.optimize import OptimizeResult
from zquantum.core.history.recorder import HistoryEntry
from zquantum.core.history.recorder import recorder as _recorder
from zquantum.core.interfaces.ansatz import Ansatz
from zquantum.core.interfaces.cost_function import CostFunction
from zquantum.core.interfaces.optimizer import (
    NestedOptimizer,
    Optimizer,
    extend_histories,
)
from zquantum.core.typing import RecorderFactory


class FourierOptimizer(NestedOptimizer):
    @property
    def inner_optimizer(self) -> Optimizer:
        return self._inner_optimizer

    @property
    def recorder(self) -> RecorderFactory:
        return self._recorder

    def __init__(
        self,
        ansatz: Ansatz,
        inner_optimizer: Optimizer,
        min_layer: int,
        max_layer: int,
        n_layers_per_iteration: int = 1,
        q: Union[int, float] = np.inf,
        R: int = 10,
        recorder: RecorderFactory = _recorder,
    ) -> None:
        """TODO: description of fourier from the other branch + example use case?
        Args:
            ansatz: ansatz that will be used for creating the cost function.
            inner_optimizer: optimizer used for optimization of parameters
                after adding a new layer to the ansatz.
            min_layer: starting number of layers.
            max_layer: maximum number of layers, at which optimization should stop.
            n_layers_per_iteration: number of layers added for each iteration.
            q: length of each of u and v parameters. Can be any positive integer or
                infinity. If q = infinity, then q = n_layers and grows unbounded.
            R: the number of random perturbations we add to the parameters so that we
                can sometimes escape a local optimum. Can be any integer 0 or above. See
                paragraph 2 of Appendix B2 in the original paper for more details.

        """

        assert 0 < min_layer < max_layer
        assert n_layers_per_iteration > 0
        assert q > 0
        assert isinstance(q, int) or q == np.inf
        assert R >= 0
        self._ansatz = ansatz
        self._inner_optimizer = inner_optimizer
        self._min_layer = min_layer
        self._max_layer = max_layer
        self._n_layers_per_iteration = n_layers_per_iteration
        self._recorder = recorder
        self._q = q
        self._R = R

    def _minimize(
        self,
        cost_function_factory: Callable[[Ansatz], CostFunction],
        initial_params: np.ndarray,
        keep_history: bool = False,
    ) -> OptimizeResult:
        """Finds the parameters that minimize the value of the cost function created
        using `cost_function_factory`. In each iteration, the number of layers of ansatz
        are increased, and therefore new cost function is created and the size of gammas
        and betas increases.
        NOTE:
            - The returned optimal parameters are u and v. If you wish to convert them
                to gamma and beta, use the `convert_u_v_to_gamma_beta` function.
            - The returned optimal parameters, when converted to gamma and beta, should
                minimize the value of the cost function for the ansatz with number of
                layers specified by `max_layer`.
            - The returned `nit` is total number of iterations from each call the inner
                optimizer combined.

        Args:
            cost_function_factory: a function that returns a cost function that depends
                on the provided ansatz.
            inital_params: initial parameters u and v. Should be a 1d array of size
                `q * 2`. Or, if q = infinity, it should be of size `min_layer`.
            keep_history: flag indicating whether history of cost function
                evaluations should be recorded.

        """
        self._validate_initial_params(initial_params)
        ansatz = copy.deepcopy(self._ansatz)

        nit = 0
        nfev = 0
        histories: Dict[str, List[HistoryEntry]] = defaultdict(list)
        histories["history"] = []

        for n_layers in range(
            self._min_layer, self._max_layer + 1, self._n_layers_per_iteration
        ):
            # Setup for new layer
            ansatz.number_of_layers = n_layers
            gamma_beta_cost_function = cost_function_factory(ansatz)

            def u_v_cost_function(parameters: np.ndarray) -> float:
                gamma_beta = convert_u_v_to_gamma_beta(n_layers, parameters)
                return gamma_beta_cost_function(gamma_beta)

            if keep_history:
                u_v_cost_function = self.recorder(u_v_cost_function)

            # Setup new initial params and start optimization
            best_value_this_layer = np.inf
            if n_layers == self._min_layer:
                opt_unperturbed_u_v = initial_params
            else:
                # Increment the length of u and v if q = infinity
                if self._q == np.inf:
                    opt_unperturbed_u_v = self._get_u_v_for_next_layer(
                        opt_unperturbed_u_v
                    )
                    best_u_v_so_far = self._get_u_v_for_next_layer(best_u_v_so_far)

                # Perturb u and v from the best u and v so far when incrementing a
                # layer, as demonstrated in figure 10
                if self._R > 0:
                    all_r_plus_1_perturbed_params = [
                        _perturb_params_randomly(best_u_v_so_far)
                        for _ in range(self._R)
                    ]
                    all_r_plus_1_perturbed_params.append(best_u_v_so_far)

                    # Optimize perturbed u and v. There are `self._R + 1` number of
                    # perturbed parameters to optimize separtely, as indicated in figure
                    # 10.
                    for perturbed_params in all_r_plus_1_perturbed_params:
                        local_results = self.inner_optimizer.minimize(
                            u_v_cost_function, perturbed_params, keep_history=False
                        )
                        nfev += local_results.nfev
                        nit += local_results.nit
                        if local_results.opt_value < best_value_this_layer:
                            best_value_this_layer = local_results.opt_value
                            best_u_v_so_far = local_results.opt_params

            # Optimize unperturbed u and v
            layer_results = self.inner_optimizer.minimize(
                u_v_cost_function, opt_unperturbed_u_v, keep_history=False
            )

            # Best u_v so far is passed onto the next layer, along with optimized
            # unperturbed u_v, as indicated in the caption of figure 10.
            opt_unperturbed_u_v = layer_results.opt_params
            if layer_results.opt_value < best_value_this_layer:
                best_value_this_layer = layer_results.opt_value
                best_u_v_so_far = layer_results.opt_params

            nfev += layer_results.nfev
            nit += layer_results.nit

            if keep_history:
                histories = extend_histories(u_v_cost_function, histories)

        del layer_results["history"]
        del layer_results["nit"]
        del layer_results["nfev"]
        del layer_results["opt_params"]

        return OptimizeResult(
            **layer_results, **histories, nfev=nfev, nit=nit, opt_params=best_u_v_so_far
        )

    def _validate_initial_params(self, initial_params: np.ndarray):
        if len(initial_params.shape) != 1:
            raise ValueError("Initial params should be a 1d array.")
        elif self._q == np.inf and initial_params.size != self._min_layer * 2:
            raise ValueError(
                "When q = infinity, initial params should of size min_layer * 2."
            )
        elif self._q != np.inf and initial_params.size != self._q * 2:
            raise ValueError("Initial params should of size q * 2.")

    def _get_u_v_for_next_layer(self, u_v: np.ndarray) -> np.ndarray:
        """When q = infinity, u_v is extended at the increment of each layer such that
        the length of u and v is equal to length of gamma and beta. See equation B3
        of the original paper.

        """
        return np.append(u_v, np.zeros(2 * self._n_layers_per_iteration))


def convert_u_v_to_gamma_beta(n_layers, u_v: np.ndarray) -> np.ndarray:
    """Performs a "Discrete Sine/Cosine Transform" to convert u and v parameters into
    gamma and beta. See equation B2 of the original paper for how this is done.

    Args:
        n layers is for size of output gamma/beta params.
        u_v: parameters u and v in a 1d array with `u` ordered before `v`
    Returns:
        parameters gamma and beta in a 1d array of size `2 * n_layers`
    """
    assert n_layers > 0
    if not len(u_v.shape) == 1:
        raise ValueError("Parameters must be a 1d array.")
    if not u_v.size % 2 == 0:
        raise ValueError("Size of parameters must be even.")

    # input parameters are u and v parameters of size 2q.
    q = u_v.size // 2
    u = u_v.reshape(-1, 2).T[0]
    v = u_v.reshape(-1, 2).T[1]

    # Calculate gamma of each layer given `u` and
    # Calculate beta of each layer given `v`,
    # see eq (B2) of original paper.
    gammas_and_betas = []
    for i in range(n_layers):
        gamma = u.dot(np.sin(np.pi / n_layers * (np.arange(q) + 0.5) * (i + 0.5)))
        beta = v.dot(np.cos(np.pi / n_layers * (np.arange(q) + 0.5) * (i + 0.5)))
        gammas_and_betas.append(gamma)
        gammas_and_betas.append(beta)

    # output parameters are of size (2 * n_layers).
    assert len(gammas_and_betas) == n_layers * 2
    return np.array(gammas_and_betas)


def _perturb_params_randomly(u_v: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Performs 1 random perturbation. See Equation B5.

    Alpha is a free parameter corresponding to the strength of the perturbation. A value
    of 0.6 is what was found to work best by the authors of the original paper and is
    what we use in this implementation of Fourier.
    """
    stdev = np.sqrt(np.abs(u_v))  # u_v is variance
    return u_v + alpha * np.random.normal(0, stdev)
