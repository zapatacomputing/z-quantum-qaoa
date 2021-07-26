from zquantum.core.symbolic_simulator import SymbolicSimulator
from zquantum.qaoa.ansatzes.farhi_ansatz import QAOAFarhiAnsatz
from zquantum.qaoa.recursive_qaoa import RecursiveQAOA
from zquantum.core.estimation import (
    estimate_expectation_values_by_averaging,
    allocate_shots_uniformly,
    calculate_exact_expectation_values,
)
import numpy as np

from zquantum.qaoa.problems import (
    get_maxcut_hamiltonian,
)
import networkx as nx
from zquantum.core.openfermion import change_operator_type

from openfermion import IsingOperator
from functools import partial
from zquantum.optimizers import BasinHoppingOptimizer
import pytest


class TestMaxcut:
    @pytest.fixture()
    def hamiltonian(self):
        return (
            IsingOperator("Z0 Z1", 5)
            + IsingOperator("Z0 Z3", 5)
            + IsingOperator("Z1 Z2", 0.5)
            + IsingOperator("Z2 Z3", 0.5)
        )

    @pytest.fixture()
    def backend(self):
        return SymbolicSimulator()

    def test_rqaoa_returns_correct_answer(self, hamiltonian, backend):
        # Given

        hmph = RecursiveQAOA(hamiltonian)

        np.random.seed(43)
        estimation_preprocessors = []
        estimation_method = calculate_exact_expectation_values

        RQAOA_answers = []
        optimizer = BasinHoppingOptimizer()

        G = nx.Graph()
        G.add_nodes_from([0, 1, 2, 3])
        G.add_edge(0, 1, weight=10)
        G.add_edge(0, 3, weight=10)
        G.add_edge(1, 2, weight=1)
        G.add_edge(2, 3, weight=1)
        H = get_maxcut_hamiltonian(G)
        H = change_operator_type(H, IsingOperator)

        answers = hmph(
            n_c=3,
            ansatz=QAOAFarhiAnsatz,
            n_layers=2,
            estimation_method=estimation_method,
            estimation_preprocessors=estimation_preprocessors,
            initial_params=np.random.rand(4) * np.pi,
            optimizer=optimizer,
            n_samples=10000,
            backend=backend,
        )
        # for answer in answers:

        #     RQAOA_answers.append(answer)

        # counter = Counter(RQAOA_answers)

        n_qubits = 4
        for answer in answers:
            assert len(answer) == n_qubits

        assert answers == [(1, 0, 1, 0), (0, 1, 0, 1)] or answers == [
            (0, 1, 0, 1),
            (1, 0, 1, 0),
        ]

        # self.assertGreater(counter[(1, 0, 1, 0)], counter[(0, 0, 0, 1)])
        # self.assertGreater(counter[(0, 1, 0, 1)], counter[(0, 1, 1, 1)])

    def test_rqaoa_returns_correct_answer_recursively(self, hamiltonian, backend):
        # Given

        hmph = RecursiveQAOA(hamiltonian)

        np.random.seed(43)

        shot_allocation = partial(allocate_shots_uniformly, number_of_shots=10000)
        estimation_preprocessors = [shot_allocation]
        estimation_method = estimate_expectation_values_by_averaging

        RQAOA_answers = []
        optimizer = BasinHoppingOptimizer()
        # for j in range(10):
        answers = hmph(
            n_c=2,
            ansatz=QAOAFarhiAnsatz,
            n_layers=2,
            estimation_method=estimation_method,
            estimation_preprocessors=estimation_preprocessors,
            initial_params=np.random.rand(4) * np.pi,
            optimizer=optimizer,
            n_samples=10000,
            backend=backend,
        )
        # for answer in answers:

        #     RQAOA_answers.append(answer)
        breakpoint()

        # counter = Counter(RQAOA_answers)

        assert answers == [(1, 0, 1, 0), (0, 1, 0, 1)] or answers == [
            (0, 1, 0, 1),
            (1, 0, 1, 0),
        ]

        # self.assertGreater(counter[(1, 0, 1, 0)], counter[(0, 0, 0, 1)])
        # self.assertGreater(counter[(0, 1, 0, 1)], counter[(0, 1, 1, 1)])
