################################################################################
# © Copyright 2021 Zapata Computing Inc.
################################################################################
import networkx as nx


def make_graph(node_ids, edges, use_weights=False):
    graph = nx.Graph()
    graph.add_nodes_from(node_ids)
    if use_weights:
        graph.add_weighted_edges_from(edges)
    else:
        graph.add_edges_from(edges)
    return graph


def graph_node_index(graph, node_id):
    return next(node_i for node_i, node in enumerate(graph.nodes) if node == node_id)
