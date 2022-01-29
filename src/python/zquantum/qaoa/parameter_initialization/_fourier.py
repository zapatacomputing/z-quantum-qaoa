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


def append_random_params(target_size: int, params: np.ndarray) -> np.ndarray:
    """
    Adds new random parameters to the `params` so that the size
    of the output is `target_size`.
    New parameters are sampled from a uniform distribution over [-pi, pi].

    Args:
        target_size: target number of parameters
        params: params that we want to extend
    """
    assert len(params) < target_size
    new_params = np.random.uniform(-np.pi, np.pi, target_size - len(params))
    return np.concatenate([params, new_params])


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
        q: Union[int, np.inf] = np.inf,
        r: int = 10,
        n_layers_per_iteration: int = 1,
        recorder: RecorderFactory = _recorder,
    ) -> None:
        """
        Args:
            ansatz: ansatz that will be used for creating the cost function.
            inner_optimizer: optimizer used for optimization of parameters
                after adding a new layer to the ansatz.
            min_layer: starting number of layers.
            max_layer: maximum number of layers, at which optimization should stop.
            n_layers_per_iteration: number of layers added for each iteration.
            parameters_initializer: method for initializing parameters of the added layers.
                See append_new_random_params for example of an implementation.
        """

        assert 0 <= min_layer <= max_layer
        assert n_layers_per_iteration > 0
        assert q > 0
        assert r >= 0
        self._ansatz = ansatz
        self._inner_optimizer = inner_optimizer
        self._min_layer = min_layer
        self._max_layer = max_layer
        self._n_layers_per_iteration = n_layers_per_iteration
        self._recorder = recorder
        self._q = q
        self._r = r

    def _minimize(
        self,
        cost_function_factory: Callable[[Ansatz], CostFunction],
        initial_params: np.ndarray,
        keep_history: bool = False,
    ) -> OptimizeResult:
        """Main loop across layers.
        NOTE:
            - The optimal parameters should minimize the value of the cost function for the ansatz with
                number of layers specified by `max_layer`.

        Args:
            cost_function_factory: a function that returns a cost function that depends on the provided ansatz.
            inital_params: initial parameters u and v. Should be of size min_layer if q = infinity else q
            keep_history: flag indicating whether history of cost function
                evaluations should be recorded.

        """
        # TODO: we are returning the optimal parameters as u/v or gamma/beta? u/v kinda useless to user
        self._validate_initial_params(initial_params)
        ansatz = copy.deepcopy(self._ansatz)
        ansatz.number_of_layers = self._min_layer

        nit = 0
        nfev = 0
        histories: Dict[str, List[HistoryEntry]] = defaultdict(list)
        histories["history"] = []

        for n_layers in range(
            self._min_layer, self._max_layer + 1, self._n_layers_per_iteration
        ):
            print("Optimizing layer", n_layers)
            # Setup for new layer
            ansatz.number_of_layers += self._n_layers_per_iteration
            assert ansatz.number_of_layers == n_layers
            gamma_beta_cost_function = cost_function_factory(ansatz)

            def u_v_cost_function(parameters: np.ndarray) -> float:
                gamma_beta = _convert_u_v_to_gamma_beta(n_layers, parameters)
                # We're not missing store_artifact argument, it is optional in
                # `CallableWithArtifacts`.
                return gamma_beta_cost_function(gamma_beta)  # type: ignore

            # Setup new initial params
            if n_layers == self._min_layer:
                initial_u_v_this_iteration = initial_params
            else:
                if self._q == np.inf:
                    initial_u_v_this_iteration = self._get_u_v_for_next_layer(
                        optimal_u_v
                    )
                else:
                    initial_u_v_this_iteration = optimal_u_v

            if keep_history:
                # compatible function signature in assignment.
                u_v_cost_function = self.recorder(u_v_cost_function)  # type: ignore

            # Optimize
            layer_results = self.inner_optimizer.minimize(
                u_v_cost_function, initial_u_v_this_iteration, keep_history=False
            )
            optimal_u_v: np.ndarray = layer_results.opt_params

            nfev += layer_results.nfev
            nit += layer_results.nit

            if keep_history:
                # If keep_history then u_v_cost_function will be a recorder and not
                # just a cost function.
                histories = extend_histories(u_v_cost_function, histories)  # type: ignore

        del layer_results["history"]
        del layer_results["nit"]
        del layer_results["nfev"]

        return OptimizeResult(**layer_results, **histories, nfev=nfev, nit=nit)

    def _validate_initial_params(self, initial_params: np.ndarray):
        is_valid = len(initial_params.shape) == 1
        if self._q == np.inf:
            is_valid = is_valid and initial_params.size == self._min_layer
        else:
            is_valid = is_valid and initial_params.size == self._q
        if not is_valid:
            raise ValueError(
                "Initial params should be a 1d array of size min_layer if q = infinity else q"
            )

    def _get_u_v_for_next_layer(self, u_v: np.ndarray) -> np.ndarray:
        """Equation B3
        For when q = infinity

        """
        return np.append(u_v, np.zeros(2 * self._n_layers_per_iteration))


def _convert_u_v_to_gamma_beta(n_layers, u_v: np.ndarray) -> np.ndarray:
    """Equation B2
    Args:
        n layers is for size of output gamma/beta params.
        u_v: parameters u and v in a 1d array with `u` before `v`
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
