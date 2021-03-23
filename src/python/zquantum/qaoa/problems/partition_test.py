import networkx as nx
import pytest

from .partition import get_graph_partition_hamiltonian


def _graph_from_edges(edges):
    graph = nx.Graph()
    for edge in edges:
        graph.add_edge(*edge)
    return graph


GRAPH_OPERATOR_TERM_PAIRS = [
    (
        _graph_from_edges([(0, 1)]),
        {
            (): 2.5+0j,
            ((0, 'Z'), (1, 'Z')): 1.5+0j,
        }
    ),
    (
        _graph_from_edges([(0, 1), (0, 2)]),
        {
            (): 4+0j,
            ((0, 'Z'), (1, 'Z')): 1.5+0j,
            ((0, 'Z'), (2, 'Z')): 1.5+0j,
            ((1, 'Z'), (2, 'Z')): 2.0+0j,
        }
    ),
    (
        _graph_from_edges([(0, 1), (0, 2), (0, 3)]),
        {
            (): 5.5+0j,
            ((0, 'Z'), (1, 'Z')): 1.5+0j,
            ((0, 'Z'), (2, 'Z')): 1.5+0j,
            ((0, 'Z'), (3, 'Z')): 1.5+0j,
            ((1, 'Z'), (2, 'Z')): 2+0j,
            ((1, 'Z'), (3, 'Z')): 2+0j,
            ((2, 'Z'), (3, 'Z')): 2+0j,
        }
    ),
    (
        _graph_from_edges([(0, 1), (1, 2), (3, 4)]),
        {
            (): 6.5+0j,
            ((0, 'Z'), (1, 'Z')): 1.5+0j,
            ((0, 'Z'), (2, 'Z')): 2+0j,
            ((0, 'Z'), (3, 'Z')): 2+0j,
            ((0, 'Z'), (4, 'Z')): 2+0j,
            ((1, 'Z'), (2, 'Z')): 1.5+0j,
            ((1, 'Z'), (3, 'Z')): 2+0j,
            ((1, 'Z'), (4, 'Z')): 2+0j,
            ((2, 'Z'), (3, 'Z')): 2+0j,
            ((2, 'Z'), (4, 'Z')): 2+0j,
            ((3, 'Z'), (4, 'Z')): 1.5+0j,
        }
    )
]


class TestGetHamiltonian:
    @pytest.mark.parametrize('graph,terms', GRAPH_OPERATOR_TERM_PAIRS)
    def test_returns_expected_terms(self, graph, terms):
        qubit_operator = get_graph_partition_hamiltonian(graph)
        assert qubit_operator.terms == terms
