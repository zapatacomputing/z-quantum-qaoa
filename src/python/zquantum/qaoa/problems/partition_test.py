import networkx as nx
import pytest
# from zquantum.qaoa.farhi_ansatz import QAOAFarhiAnsatz

from .partition import get_graph_partition_hamiltonian


def _make_graph(adj_dict):
    graph = nx.Graph()
    for src_node, dest_nodes in adj_dict.items():
        for dest_node in dest_nodes:
            graph.add_edge(src_node, dest_node)
    return graph


def _graph_from_edges(edges):
    graph = nx.Graph()
    for edge in edges:
        graph.add_edge(*edge)
    return graph


GRAPH_OPERATOR_TERM_PAIRS = [
    (
        _make_graph({0: [1]}),
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
        _make_graph({0: [1, 2, 3]}),
        {
            (): 5.5+0j,
            ((0, 'Z'), (1, 'Z')): 1.5+0j,
            ((0, 'Z'), (2, 'Z')): 1.5+0j,
            ((0, 'Z'), (3, 'Z')): 1.5+0j,
            ((1, 'Z'), (2, 'Z')): 2+0j,
            ((1, 'Z'), (3, 'Z')): 2+0j,
            ((2, 'Z'), (3, 'Z')): 2+0j,
        }
    )
]


class TestGetHamiltonian:
    @pytest.mark.parametrize('graph,terms', GRAPH_OPERATOR_TERM_PAIRS)
    def test_returns_expected_terms(self, graph, terms):
        qubit_operator = get_graph_partition_hamiltonian(graph)
        assert qubit_operator.terms == terms
