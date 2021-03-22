import networkx as nx
import pytest

from .partition import get_graph_partition_hamiltonian


def _make_graph(adj_dict):
    graph = nx.Graph()
    for src_node, dest_nodes in adj_dict.items():
        for dest_node in dest_nodes:
            graph.add_edge(src_node, dest_node)
    return graph


GRAPH_OPERATOR_TERM_PAIRS = [
    (
        _make_graph({0: [1]}),
        {
            (): 2.5+0j,
            ((0, 'Z'), (1, 'Z')): 1.5+0j
        }
    ),
]


class TestGetHamiltonian:
    @pytest.mark.parametrize('graph,terms', GRAPH_OPERATOR_TERM_PAIRS)
    def test_returns_expected_terms(self, graph, terms):
        qubit_operator = get_graph_partition_hamiltonian(graph)
        assert qubit_operator.terms == terms
