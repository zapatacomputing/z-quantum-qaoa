import networkx as nx
from zquantum.qaoa.farhi_ansatz import QAOAFarhiAnsatz

from .partition import get_graph_partition_hamiltonian

class TestGetHamiltonian:
    def test_sanity(self):
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2, 3])
        G.add_edge(0, 1, weight=10)
        G.add_edge(0, 3, weight=10)
        G.add_edge(1, 2, weight=1)
        G.add_edge(2, 3, weight=1)
        qubit_operator = get_graph_partition_hamiltonian(G)
        assert qubit_operator is None

