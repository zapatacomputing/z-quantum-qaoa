################################################################################
# © Copyright 2021 Zapata Computing Inc.
################################################################################
import copy

import networkx as nx
import pytest
from zquantum.qaoa.problems import GraphPartitioning

from ._helpers import graph_node_index, make_graph

MONOTONIC_GRAPH_OPERATOR_TERM_PAIRS = [
    (
        make_graph(node_ids=range(2), edges=[(0, 1)]),
        {
            (): 2.5,
            ((0, "Z"), (1, "Z")): 1.5,
        },
    ),
    (
        make_graph(node_ids=range(3), edges=[(0, 1), (0, 2)]),
        {
            (): 4,
            ((0, "Z"), (1, "Z")): 1.5,
            ((0, "Z"), (2, "Z")): 1.5,
            ((1, "Z"), (2, "Z")): 2.0,
        },
    ),
    (
        make_graph(node_ids=range(4), edges=[(0, 1), (0, 2), (0, 3)]),
        {
            (): 5.5,
            ((0, "Z"), (1, "Z")): 1.5,
            ((0, "Z"), (2, "Z")): 1.5,
            ((0, "Z"), (3, "Z")): 1.5,
            ((1, "Z"), (2, "Z")): 2,
            ((1, "Z"), (3, "Z")): 2,
            ((2, "Z"), (3, "Z")): 2,
        },
    ),
    (
        make_graph(node_ids=range(5), edges=[(0, 1), (1, 2), (3, 4)]),
        {
            (): 6.5,
            ((0, "Z"), (1, "Z")): 1.5,
            ((0, "Z"), (2, "Z")): 2,
            ((0, "Z"), (3, "Z")): 2,
            ((0, "Z"), (4, "Z")): 2,
            ((1, "Z"), (2, "Z")): 1.5,
            ((1, "Z"), (3, "Z")): 2,
            ((1, "Z"), (4, "Z")): 2,
            ((2, "Z"), (3, "Z")): 2,
            ((2, "Z"), (4, "Z")): 2,
            ((3, "Z"), (4, "Z")): 1.5,
        },
    ),
]

GRAPH_OPERATOR_TERM_SCALING_OFFSET_LIST = [
    (
        make_graph(node_ids=range(4), edges=[(0, 1), (0, 2), (0, 3)]),
        {
            (): 18,
            ((0, "Z"), (1, "Z")): 3,
            ((0, "Z"), (2, "Z")): 3,
            ((0, "Z"), (3, "Z")): 3,
            ((1, "Z"), (2, "Z")): 4,
            ((1, "Z"), (3, "Z")): 4,
            ((2, "Z"), (3, "Z")): 4,
        },
        2.0,
        7.0,
    ),
]

NONMONOTONIC_GRAPH_OPERATOR_TERM_PAIRS = [
    (
        make_graph(node_ids=[4, 2], edges=[(2, 4)]),
        {
            (): 2.5,
            ((0, "Z"), (1, "Z")): 1.5,
        },
    ),
    (
        make_graph(node_ids="CBA", edges=[("C", "B"), ("C", "A")]),
        {
            (): 4,
            ((0, "Z"), (1, "Z")): 1.5,  # the C-B edge
            ((0, "Z"), (2, "Z")): 1.5,  # the C-A edge
            ((1, "Z"), (2, "Z")): 2.0,  # the B-C edge
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
    (make_graph(node_ids=range(2), edges=[(0, 1)]), [0, 0], 4),
    (make_graph(node_ids=range(2), edges=[(0, 1)]), [0, 1], 1),
    (make_graph(node_ids=range(4), edges=[(0, 1), (0, 2), (0, 3)]), [0, 0, 0, 0], 16),
    (make_graph(node_ids=range(4), edges=[(0, 1), (0, 2), (0, 3)]), [0, 0, 1, 1], 2),
    (make_graph(node_ids=range(4), edges=[(0, 1), (0, 2), (0, 3)]), [0, 1, 1, 1], 7),
    (
        make_graph(node_ids=range(5), edges=[(0, 1), (1, 2), (3, 4)]),
        [1, 1, 1, 1, 1],
        25,
    ),
]

GRAPH_BEST_SOLUTIONS_COST_LIST = [
    (make_graph(node_ids=range(2), edges=[(0, 1)]), [(0, 1), (1, 0)], 1),
    (
        make_graph(node_ids=range(3), edges=[(0, 1), (0, 2)]),
        [(1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 0)],
        2,
    ),
    (
        make_graph(node_ids=range(4), edges=[(0, 1), (0, 2), (0, 3)]),
        [
            (0, 0, 1, 1),
            (0, 1, 0, 1),
            (0, 1, 1, 0),
            (1, 0, 0, 1),
            (1, 0, 1, 0),
            (1, 1, 0, 0),
        ],
        2,
    ),
    (
        make_graph(node_ids=range(5), edges=[(0, 1), (1, 2), (3, 4)]),
        [(0, 0, 0, 1, 1), (1, 1, 1, 0, 0)],
        1,
    ),
]


class TestGetGraphPartitionHamiltonian:
    @pytest.mark.parametrize(
        "graph,terms",
        [
            *MONOTONIC_GRAPH_OPERATOR_TERM_PAIRS,
            *NONMONOTONIC_GRAPH_OPERATOR_TERM_PAIRS,
        ],
    )
    def test_returns_expected_terms(self, graph, terms):
        qubit_operator = GraphPartitioning().get_hamiltonian(graph)
        assert qubit_operator.terms == terms

    @pytest.mark.parametrize(
        "graph,terms,scale_factor,offset",
        [*GRAPH_OPERATOR_TERM_SCALING_OFFSET_LIST],
    )
    def test_scaling_and_offset_works(self, graph, terms, scale_factor, offset):
        qubit_operator = GraphPartitioning().get_hamiltonian(
            graph, scale_factor, offset
        )
        assert qubit_operator.terms == terms

    @pytest.mark.parametrize("graph", GRAPH_EXAMPLES)
    def test_has_1_5_weight_on_edge_terms(self, graph: nx.Graph):
        qubit_operator = GraphPartitioning().get_hamiltonian(graph)

        for vertex_id1, vertex_id2 in graph.edges:
            qubit_index1 = graph_node_index(graph, vertex_id1)
            qubit_index2 = graph_node_index(graph, vertex_id2)
            assert (
                qubit_operator.terms[((qubit_index1, "Z"), (qubit_index2, "Z"))] == 1.5
            )


class TestEvaluateGraphPartitionSolution:
    @pytest.mark.parametrize("graph,solution,target_value", [*GRAPH_SOLUTION_COST_LIST])
    def test_evaluate_graph_partition_solution(self, graph, solution, target_value):
        value = GraphPartitioning().evaluate_solution(solution, graph)
        assert value == target_value

    @pytest.mark.parametrize("graph,solution,target_value", [*GRAPH_SOLUTION_COST_LIST])
    def test_evaluate_graph_partition_solution_with_invalid_input(
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
                _ = GraphPartitioning().evaluate_solution(invalid_solution, graph)


class TestSolveGraphPartitionByExhaustiveSearch:
    @pytest.mark.parametrize(
        "graph,target_solutions,target_value", [*GRAPH_BEST_SOLUTIONS_COST_LIST]
    )
    def test_solve_graph_partition_by_exhaustive_search(
        self, graph, target_solutions, target_value
    ):
        value, solutions = GraphPartitioning().solve_by_exhaustive_search(graph)
        assert set(solutions) == set(target_solutions)
        assert value == target_value
