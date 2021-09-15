import numpy as np
import copy
from cvxopt import matrix, solvers
from .utils import hyperplane_rounding, z_vec_to_x_vec

solvers.options['show_progress'] = False

# This file contains an implementation of the Karloff-Zwick algorithm for MAX 3-SAT from
# "A 7=8-Approximation Algorithm for MAX 3SAT?", IEEE FOCS 1997

def solve_kz_sdp_for_3_sat(three_sat):
    n = three_sat.num_variables
    m = three_sat.num_clauses
    num_constraints = 4 * m + 3 * n + 1

    c = np.ones(num_constraints)
    c = matrix(c)

    G0 = np.zeros((4*m, num_constraints))
    G0[:, 3*n+1:] = -np.eye(4*m)
    G0 = matrix(G0)

    h0 = np.zeros((4*m, 1))
    h0 = matrix(h0)

    h = np.zeros((2*n+1, 2*n+1))
    hs = [matrix(h)]

    G = np.zeros(((2*n+1)*(2*n+1), num_constraints))
    for i in range(2*n+1):
        G[i*(2*n+1)+i, i] = -1.0
    for i in range(n):
        G[2*i*(2*n+1)+2*i+1, 2*n+1+i] = 1.0

    A = np.zeros((m, num_constraints))
    b = np.zeros((m, 1))

    idx = 3 * n + 1
    idx2 = 0
    for clause, weight in zip(three_sat.clauses, three_sat.weights):
        if len(clause) == 3:
            i, j, k = list(clause)
        elif len(clause) == 2:
            i, j = list(clause)
            k = 2 * n
        else:
            i = list(clause)[0]
            j = 2 * n
            k = 2 * n

        G[2*n*(2*n+1)+j, idx] = -0.25
        G[2*n*(2*n+1)+k, idx] = -0.25
        G[i*(2*n+1)+j, idx] = -0.25
        G[i*(2*n+1)+k, idx] = -0.25
        A[idx2, idx] = -1.0

        G[2*n*(2*n+1)+i, idx+1] = -0.25
        G[2*n*(2*n+1)+k, idx+1] = -0.25
        G[j*(2*n+1)+i, idx+1] = -0.25
        G[j*(2*n+1)+k, idx+1] = -0.25
        A[idx2, idx+1] = -1.0

        G[2*n*(2*n+1)+i, idx+2] = -0.25
        G[2*n*(2*n+1)+j, idx+2] = -0.25
        G[k*(2*n+1)+i, idx+2] = -0.25
        G[k*(2*n+1)+j, idx+2] = -0.25
        A[idx2, idx+2] = -1.0

        A[idx2, idx+3] = -1.0

        b[idx2, :] = -weight

        idx += 4
        idx2 += 1

    Gs = [matrix(G)]
    A = matrix(A)
    b = matrix(b)

    sol = solvers.sdp(c, Gl=G0, hl=h0, Gs=Gs, hs=hs, A=A, b=b)
    opt_z = np.array(sol['zs'][0])
    opt_x = np.array(sol['x']).squeeze()

    E_z, V_z = np.linalg.eig(opt_z[::2, ::2])
    E_z = np.maximum(E_z, 1e-30)
    vectors = np.dot(V_z, np.diag(E_z)**0.5)

    return opt_z, vectors, opt_x

def solve_max_3_sat_by_kz_alg(three_sat, num_roundings=10):
    _, vectors, _ = solve_kz_sdp_for_3_sat(three_sat)
    assignments = []
    values = []
    for _ in range(num_roundings):
        z_vec = hyperplane_rounding(vectors, last_entry=1)[:-1]
        assignment = z_vec_to_x_vec(z_vec)
        value = three_sat.evaluate_assignment(assignment)
        assignments.append(assignment)
        values.append(value)
    return assignments, values
