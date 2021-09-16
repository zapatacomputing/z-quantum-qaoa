import networkx as nx
import numpy as np
import random
from openfermion import IsingOperator
from .utils import get_x_vec, get_z_vec

def generate_random_graph(topology_specs, weight_specs):
    if topology_specs['type'] == 'gnp':
        G = nx.gnp_random_graph(topology_specs['num_nodes'], topology_specs['edge_prob'])
    elif topology_specs['type'] == 'gnm':
        G = nx.gnm_random_graph(topology_specs['num_nodes'], topology_specs['num_edges'])
    elif topology_specs['type'] == 'regular':
        G = nx.generators.random_graphs.random_regular_graph(topology_specs['degree'], topology_specs['num_nodes'])
    else:
        G = nx.complete_graph(topology_specs['num_nodes'])

    if weight_specs['type'] == 'discrete':
        weighted_edges = [(e[0], e[1], random.choice(weight_specs['candidates'])) for e in G.edges]
    elif weight_specs['type'] == 'uniform':
        weighted_edges = [(e[0], e[1], np.random.uniform(weight_specs['low'], weight_specs['high'], 1)[0]) for e in G.edges]
    elif weight_specs['type'] == 'normal':
        weighted_edges = [(e[0], e[1], np.random.normal(weight_specs['loc'], weight_specs['scale'], 1)[0]) for e in G.edges]
    else:
        weighted_edges = [(e[0], e[1], 1.0) for e in G.edges]

    G.add_weighted_edges_from(weighted_edges)
    return G

def get_all_edges_of_a_graph(graph):
    n = graph.number_of_nodes()
    laplacian = nx.laplacian_matrix(graph).todense().real
    laplacian = np.array(laplacian)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if laplacian[i, j] != 0:
                edges.append((i, j, -laplacian[i, j]))
    return edges

def construct_neighbor_dictionary(graph):
    n = graph.number_of_nodes()
    adjacency_matrix = np.array(nx.adjacency_matrix(graph).todense().real)
    neighbors = {}
    for i in range(n):
        neighbors[i] = set()
        for j in range(n):
            if i == j:
                continue
            if adjacency_matrix[i, j] != 0:
                neighbors[i].add(j)
    return neighbors

def get_cut_size(laplacian, cut):
    return 0.25 * np.dot(cut, np.dot(laplacian, cut))

def get_all_cut_sizes_of_a_graph(graph, use_laplacian=False, duplicate=False):
    # Slow for large graphs!
    n = graph.number_of_nodes()
    if use_laplacian:
        laplacian = nx.laplacian_matrix(graph).todense().real
        laplacian = np.array(laplacian)
        cut_sizes = []
        N = 2**n if duplicate else 2**(n-1)
        for i in range(N):
            cut = get_z_vec(i, n=n)
            cut_size = get_cut_size(laplacian, cut)
            cut_sizes.append(cut_size)
    else:
        edges = get_all_edges_of_a_graph(graph)
        cut_sizes = []
        N = 2**n if duplicate else 2**(n-1)
        for i in range(N):
            cut = get_z_vec(i, n=n)
            cut_size = get_k_cut_size(edges, cut)
            cut_sizes.append(cut_size)
    return cut_sizes

def get_top_cuts_of_a_graph(graph, ratio=0.95):
    # Slow for large graphs!
    n = graph.number_of_nodes()
    cut_sizes = get_all_cut_sizes_of_a_graph(graph)
    max_cut_size = max(cut_sizes)
    min_cut_size = min(cut_sizes)
    mean_cut_size = np.mean(cut_sizes)
    threshold = mean_cut_size + (max_cut_size - mean_cut_size) * ratio
    top_cuts_and_sizes = [(get_z_vec(cut_id, n=n), cut_size)
                            for cut_id, cut_size in enumerate(cut_sizes)
                            if cut_size >= threshold]
    return top_cuts_and_sizes, max_cut_size, min_cut_size, mean_cut_size

def get_all_bisection_sizes_of_a_graph(graph, use_laplacian=False, duplicate=False):
    # Slow for large graphs!
    n = graph.number_of_nodes()
    assert n % 2 == 0
    if use_laplacian:
        laplacian = nx.laplacian_matrix(graph).todense().real
        laplacian = np.array(laplacian)
        bisection_ids = []
        bisection_sizes = []
        N = 2**n if duplicate else 2**(n-1)
        for i in range(N):
            cut = get_z_vec(i, n=n)
            if np.sum(cut) == 0:
                cut_size = get_cut_size(laplacian, cut)
                bisection_ids.append(i)
                bisection_sizes.append(cut_size)
    else:
        edges = get_all_edges_of_a_graph(graph)
        bisection_ids = []
        bisection_sizes = []
        N = 2**n if duplicate else 2**(n-1)
        for i in range(N):
            cut = get_z_vec(i, n=n)
            if np.sum(cut) == 0:
                cut_size = get_k_cut_size(edges, cut)
                bisection_ids.append(i)
                bisection_sizes.append(cut_size)
    return bisection_ids, bisection_sizes

def get_top_bisections_of_a_graph(graph, ratio=0.95):
    # Slow for large graphs!
    n = graph.number_of_nodes()
    bisection_ids, bisection_sizes = get_all_bisection_sizes_of_a_graph(graph)
    max_bisection_size = max(bisection_sizes)
    min_bisection_size = min(bisection_sizes)
    mean_bisection_size = np.mean(bisection_sizes)
    threshold = mean_bisection_size + (max_bisection_size - mean_bisection_size) * ratio
    top_bisections_and_sizes = [(get_z_vec(bisection_id, n=n), bisection_size)
                                for bisection_id, bisection_size in zip(bisection_ids, bisection_sizes)
                                if bisection_size >= threshold]
    return top_bisections_and_sizes, max_bisection_size, min_bisection_size, mean_bisection_size

def get_k_cut_size(edges, k_cut):
    res = 0.0
    for edge in edges:
        u, v, w = edge
        if k_cut[u] != k_cut[v]:
            res += w
    return res

def construct_ising_hamiltonian_for_max_cut(graph):
    n = graph.number_of_nodes()
    laplacian = nx.laplacian_matrix(graph).todense().real
    laplacian = np.array(laplacian)
    output = IsingOperator()
    for i in range(n):
        for j in range(i+1, n):
            if laplacian[i, j] != 0:
                output += IsingOperator("Z"+str(i)+" Z"+str(j), -laplacian[i, j]/2.0)
                output += IsingOperator("", laplacian[i, j]/2.0)
    return output
