import networkx as nx
import pytest
import copy

# from zquantum.qaoa.problems.stable_set import (
#     evaluate_stable_set_solution,
#     get_stable_set_hamiltonian,
#     solve_stable_set_by_exhaustive_search,
# )

from zquantum.qaoa.problems import StableSet
from ._helpers import make_graph, graph_node_index

MONOTONIC_GRAPH_OPERATOR_TERM_PAIRS = [
    (
        make_graph(node_ids=range(2), edges=[(0, 1)]),
        {
            (): -0.5,
            ((0, "Z"), (1, "Z")): 0.5,
        },
    ),
    (
        make_graph(node_ids=range(3), edges=[(0, 1), (0, 2)]),
        {
            (): -0.5,
            ((0, "Z"),): -0.5,
            ((0, "Z"), (1, "Z")): 0.5,
            ((0, "Z"), (2, "Z")): 0.5,
        },
    ),
    (
        make_graph(node_ids=range(4), edges=[(0, 1), (0, 2), (0, 3)]),
        {
            (): -0.5,
            ((0, "Z"),): -1,
            ((0, "Z"), (1, "Z")): 0.5,
            ((0, "Z"), (2, "Z")): 0.5,
            ((0, "Z"), (3, "Z")): 0.5,
        },
    ),
    (
        make_graph(node_ids=range(5), edges=[(0, 1), (1, 2), (3, 4)]),
        {
            (): -1,
            ((1, "Z"),): -0.5,
            ((0, "Z"), (1, "Z")): 0.5,
            ((1, "Z"), (2, "Z")): 0.5,
            ((3, "Z"), (4, "Z")): 0.5,
        },
    ),
]

NONMONOTONIC_GRAPH_OPERATOR_TERM_PAIRS = [
    (
        make_graph(node_ids=[4, 2], edges=[(2, 4)]),
        {
            (): -0.5,
            ((0, "Z"), (1, "Z")): 0.5,
        },
    ),
    (
        make_graph(node_ids="CBA", edges=[("C", "B"), ("C", "A")]),
        {
            (): -0.5,
            ((0, "Z"),): -0.5,
            ((0, "Z"), (1, "Z")): 0.5,
            ((0, "Z"), (2, "Z")): 0.5,
        },
    ),
]

GRAPH_EXAMPLES = [
    *[graph for graph, _ in MONOTONIC_GRAPH_OPERATOR_TERM_PAIRS],
    *[graph for graph, _ in NONMONOTONIC_GRAPH_OPERATOR_TERM_PAIRS],
    make_graph(
        node_ids=range(10),
        edges=[
            (0, 2),
            (0, 3),
            (1, 2),
            (4, 5),
            (0, 8),
        ],
    ),
    make_graph(
        node_ids=["foo", "bar", "baz"],
        edges=[
            ("foo", "baz"),
            ("bar", "baz"),
        ],
    ),
]

GRAPH_SOLUTION_COST_LIST = [
    (make_graph(node_ids=range(2), edges=[(0, 1)]), [0, 0], 0),
    (make_graph(node_ids=range(2), edges=[(0, 1)]), [0, 1], -1),
    (
        make_graph(
            node_ids=range(4), edges=[(0, 1, 1), (0, 2, 2), (0, 3, 3)], use_weights=True
        ),
        [1, 0, 0, 0],
        -1,
    ),
    (make_graph(node_ids=range(4), edges=[(0, 1), (0, 2), (0, 3)]), [0, 0, 1, 1], -2),
    (make_graph(node_ids=range(4), edges=[(0, 1), (0, 2), (0, 3)]), [0, 1, 1, 1], -3),
    (
        make_graph(node_ids=range(5), edges=[(0, 1), (1, 2), (3, 4)]),
        [1, 1, 1, 1, 1],
        1,
    ),
]

GRAPH_BEST_SOLUTIONS_COST_LIST = [
    (make_graph(node_ids=range(2), edges=[(0, 1)]), [(0, 1), (1, 0)], -1),
    (
        make_graph(node_ids=range(3), edges=[(0, 1), (0, 2)]),
        [(0, 1, 1)],
        -2,
    ),
    (
        make_graph(node_ids=range(4), edges=[(0, 1), (0, 2), (0, 3)]),
        [
            (0, 1, 1, 1),
        ],
        -3,
    ),
    (
        make_graph(node_ids=range(5), edges=[(0, 1), (1, 2), (3, 4)]),
        [(1, 0, 1, 0, 1), (1, 0, 1, 1, 0)],
        -3,
    ),
]


class TestGetStableSetHamiltonian:
    @pytest.mark.parametrize(
        "graph,terms",
        [
            *MONOTONIC_GRAPH_OPERATOR_TERM_PAIRS,
            *NONMONOTONIC_GRAPH_OPERATOR_TERM_PAIRS,
        ],
    )
    def test_returns_expected_terms(self, graph, terms):
        qubit_operator = StableSet.get_hamiltonian(graph)
        assert qubit_operator.terms == terms

    @pytest.mark.parametrize("graph", GRAPH_EXAMPLES)
    def test_has__5_weight_on_edge_terms(self, graph: nx.Graph):
        qubit_operator = StableSet.get_hamiltonian(graph)

        for vertex_id1, vertex_id2 in graph.edges:
            qubit_index1 = graph_node_index(graph, vertex_id1)
            qubit_index2 = graph_node_index(graph, vertex_id2)
            assert (
                qubit_operator.terms[((qubit_index1, "Z"), (qubit_index2, "Z"))] == 0.5
            )

    @pytest.mark.parametrize("graph", GRAPH_EXAMPLES)
    def test_has_mod__5_weight_on_vertex_terms(self, graph: nx.Graph):
        qubit_operator = StableSet.get_hamiltonian(graph)

        for vertex in graph.nodes:
            qubit_index = graph_node_index(graph, vertex)
            try:
                coefficient = qubit_operator.terms[((qubit_index, "Z"))]
            except KeyError:
                coefficient = 0
            assert coefficient % 0.5 == 0
            assert coefficient <= 0.5

    @pytest.mark.parametrize("graph", GRAPH_EXAMPLES)
    def test_has_correct_constant_term(self, graph: nx.Graph):
        expected_constant_term = 0

        qubit_operator = StableSet.get_hamiltonian(graph)
        for _ in graph.edges:
            expected_constant_term += 1 / 2

        expected_constant_term -= len(graph.nodes) / 2

        assert qubit_operator.terms[()] == expected_constant_term


class TestEvaluateStableSetSolution:
    @pytest.mark.parametrize("graph,solution,target_value", [*GRAPH_SOLUTION_COST_LIST])
    def test_evaluate_stable_set_solution(self, graph, solution, target_value):
        value = StableSet.evaluate_solution(solution, graph)
        assert value == target_value

    @pytest.mark.parametrize("graph,solution,target_value", [*GRAPH_SOLUTION_COST_LIST])
    def test_evaluate_stable_set_solution_with_invalid_input(
        self, graph, solution, target_value
    ):
        too_long_solution = solution + [1]
        too_short_solution = solution[:-1]
        invalid_value_solution = copy.copy(solution)
        invalid_value_solution[0] = -1
        invalid_solutions = [
            too_long_solution,
            too_short_solution,
            invalid_value_solution,
        ]
        for invalid_solution in invalid_solutions:
            with pytest.raises(ValueError):
                _ = StableSet.evaluate_solution(invalid_solution, graph)


class TestSolveStableSetByExhaustiveSearch:
    @pytest.mark.parametrize(
        "graph,target_solutions,target_value", [*GRAPH_BEST_SOLUTIONS_COST_LIST]
    )
    def test_solve_stable_set_by_exhaustive_search(
        self, graph, target_solutions, target_value
    ):
        value, solutions = StableSet.solve_by_exhaustive_search(graph)
        assert set(solutions) == set(target_solutions)
        assert value == target_value
