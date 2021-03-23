import networkx as nx
import pytest

from .partition import get_graph_partition_hamiltonian


def _make_graph(node_ids, edges):
    graph = nx.Graph()
    graph.add_nodes_from(node_ids)
    graph.add_edges_from(edges)
    return graph


MONOTONIC_GRAPH_OPERATOR_TERM_PAIRS = [
    (
        _make_graph(node_ids=range(2), edges=[(0, 1)]),
        {
            (): 2.5 + 0j,
            ((0, "Z"), (1, "Z")): 1.5 + 0j,
        },
    ),
    (
        _make_graph(node_ids=range(3), edges=[(0, 1), (0, 2)]),
        {
            (): 4 + 0j,
            ((0, "Z"), (1, "Z")): 1.5 + 0j,
            ((0, "Z"), (2, "Z")): 1.5 + 0j,
            ((1, "Z"), (2, "Z")): 2.0 + 0j,
        },
    ),
    (
        _make_graph(node_ids=range(4), edges=[(0, 1), (0, 2), (0, 3)]),
        {
            (): 5.5 + 0j,
            ((0, "Z"), (1, "Z")): 1.5 + 0j,
            ((0, "Z"), (2, "Z")): 1.5 + 0j,
            ((0, "Z"), (3, "Z")): 1.5 + 0j,
            ((1, "Z"), (2, "Z")): 2 + 0j,
            ((1, "Z"), (3, "Z")): 2 + 0j,
            ((2, "Z"), (3, "Z")): 2 + 0j,
        },
    ),
    (
        _make_graph(node_ids=range(5), edges=[(0, 1), (1, 2), (3, 4)]),
        {
            (): 6.5 + 0j,
            ((0, "Z"), (1, "Z")): 1.5 + 0j,
            ((0, "Z"), (2, "Z")): 2 + 0j,
            ((0, "Z"), (3, "Z")): 2 + 0j,
            ((0, "Z"), (4, "Z")): 2 + 0j,
            ((1, "Z"), (2, "Z")): 1.5 + 0j,
            ((1, "Z"), (3, "Z")): 2 + 0j,
            ((1, "Z"), (4, "Z")): 2 + 0j,
            ((2, "Z"), (3, "Z")): 2 + 0j,
            ((2, "Z"), (4, "Z")): 2 + 0j,
            ((3, "Z"), (4, "Z")): 1.5 + 0j,
        },
    ),
]

NONMONOTONIC_GRAPH_OPERATOR_TERM_PAIRS = [
    (
        _make_graph(node_ids=[4, 2], edges=[(2, 4)]),
        {
            (): 2.5 + 0j,
            ((0, "Z"), (1, "Z")): 1.5 + 0j,
        },
    ),
    (
        _make_graph(node_ids="CBA", edges=[("C", "B"), ("C", "A")]),
        {
            (): 4 + 0j,
            ((0, "Z"), (1, "Z")): 1.5 + 0j,  # the C-B edge
            ((0, "Z"), (2, "Z")): 1.5 + 0j,  # the C-A edge
            ((1, "Z"), (2, "Z")): 2.0 + 0j,  # the B-C edge
        },
    ),
]

GRAPH_EXAMPLES = [
    *[graph for graph, _ in MONOTONIC_GRAPH_OPERATOR_TERM_PAIRS],
    *[graph for graph, _ in NONMONOTONIC_GRAPH_OPERATOR_TERM_PAIRS],
    _make_graph(
        node_ids=range(10),
        edges=[
            (0, 2),
            (0, 3),
            (1, 2),
            (4, 5),
            (0, 8),
        ],
    ),
    _make_graph(
        node_ids=["foo", "bar", "baz"],
        edges=[
            ("foo", "baz"),
            ("bar", "baz"),
        ],
    ),
]


def _graph_node_index(graph, node_id):
    return next(node_i for node_i, node in enumerate(graph.nodes) if node == node_id)


class TestGetHamiltonian:
    @pytest.mark.parametrize(
        "graph,terms",
        [
            *MONOTONIC_GRAPH_OPERATOR_TERM_PAIRS,
            *NONMONOTONIC_GRAPH_OPERATOR_TERM_PAIRS,
        ],
    )
    def test_returns_expected_terms(self, graph, terms):
        qubit_operator = get_graph_partition_hamiltonian(graph)
        assert qubit_operator.terms == terms

    @pytest.mark.parametrize("graph", GRAPH_EXAMPLES)
    def test_has_1_5_weight_on_edge_terms(self, graph: nx.Graph):
        qubit_operator = get_graph_partition_hamiltonian(graph)

        for vertex_id1, vertex_id2 in graph.edges:
            qubit_index1 = _graph_node_index(graph, vertex_id1)
            qubit_index2 = _graph_node_index(graph, vertex_id2)
            assert (
                qubit_operator.terms[((qubit_index1, "Z"), (qubit_index2, "Z"))]
                == 1.5 + 0j
            )

