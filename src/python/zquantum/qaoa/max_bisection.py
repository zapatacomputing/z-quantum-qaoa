import numpy as np
import networkx as nx
from cvxopt import matrix, solvers
from utils import generate_random_vectors
from graph import get_cut_size

solvers.options['show_progress'] = False

# This file contains an implementation of the Frieze-Jerrum algorithm for max bisection from
# "Improved approximation algorithms for max k-cut and max bisection", Algorithmica, 1997

def solve_fj_sdp_for_bisection(laplacian):
    n = laplacian.shape[0]

    c = np.ones(n+1)
    c[-1] = 0
    c = matrix(c)

    G0 = np.zeros((1, n+1))
    G0[:, -1] = 1.0
    G0 = matrix(G0)

    h0 = np.zeros(1)
    h0 = matrix(h0)

    G = np.zeros((n*n, n+1))
    for i in range(n):
        G[i*(n+1), i] = -1.0
    G[:, -1] = np.ones(n*n)
    Gs = [matrix(G)]

    hs = [matrix(-laplacian)]

    sol = solvers.sdp(c, Gl=G0, hl=h0, Gs=Gs, hs=hs)
    opt_z = np.array(sol['zs'][0])
    opt_x = np.array(sol['x']).squeeze()

    E_z, V_z = np.linalg.eig(opt_z)
    E_z = np.maximum(E_z, 1e-30)
    vectors = np.dot(V_z, np.diag(E_z)**0.5)

    return opt_z, vectors, opt_x

def fj_rounding_for_bisection(vectors, laplacian):
    dim = vectors.shape[1]
    assert dim % 2 == 0

    v1, v2 = generate_random_vectors(dim, 2)
    v = v1 - v2

    S = []
    T = []
    for i in range(vectors.shape[0]):
        if np.dot(vectors[i], v)>=0:
            S.append(i)
        else:
            T.append(i)

    if len(S) < len(T):
        S, T = T, S

    S = np.array(S)
    T = np.array(T)

    weights = []
    for i in S:
        row = laplacian[i, :]
        weights.append(-np.sum(row[T]))
    ranks = np.argsort(weights)[::-1]

    S = S[ranks][:dim//2]

    res = [1 if i in S else -1 for i in range(vectors.shape[0])]
    return res

def solve_max_bisection_by_fj_alg(graph, num_roundings=10):
    n = graph.number_of_nodes()

    laplacian = nx.laplacian_matrix(graph).todense().real
    laplacian = np.array(laplacian)

    _, vectors, _ = solve_fj_sdp_for_bisection(laplacian)

    bisections = []
    sizes = []
    for _ in range(num_roundings):
        bisection = fj_rounding_for_bisection(vectors, laplacian)
        size = get_cut_size(laplacian, bisection)
        bisections.append(bisection)
        sizes.append(size)
    return bisections, sizes
