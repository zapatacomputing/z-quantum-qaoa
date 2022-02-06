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
from zquantum.qaoa.parameter_initialization import FourierOptimizer


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

        # Add `nit` entry because contract requires it.
        # This is a temporary solution, it should be changed in MockOptimizer.
        result.nit = 1

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
