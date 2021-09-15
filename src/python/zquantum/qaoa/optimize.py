import numpy as np
import scipy
import copy

def compute_cvar(dist, alpha, costs_for_cvar, ranks_for_cvar):
    total_prob = 0.0
    total_cost = 0.0
    for bitstring in ranks_for_cvar:
        prob = dist[bitstring]
        cost = costs_for_cvar[bitstring]
        if total_prob + prob < alpha:
            total_prob += prob
            total_cost += prob * cost
        else:
            total_cost += (alpha - total_prob) * cost
            break
    cvar = total_cost / alpha
    return cvar

def compute_pogs(dist, threshold, costs_for_cvar, ranks_for_cvar):
    pogs = 0.0
    for bitstring in ranks_for_cvar:
        if costs_for_cvar[bitstring] <= threshold:
            pogs += dist[bitstring]
        else:
            break
    return pogs

def get_binned_distribution(dist, num_bins, costs_for_cvar, ranks_for_cvar, max_cost, min_cost):
    binned_dist = np.zeros((num_bins))
    delta = (max_cost - min_cost) / num_bins
    for bitstring in ranks_for_cvar:
        bin_index = int((costs_for_cvar[bitstring] - min_cost) // delta)
        bin_index = min(bin_index, num_bins-1)
        binned_dist[bin_index] += dist[bitstring]
    binned_dist /= np.sum(binned_dist)
    return binned_dist

def compute_binned_cvar(grid_costs, binned_dist, alpha):
    assert len(grid_costs) == len(binned_dist)
    total_prob = 0.0
    toal_cost = 0.0
    for cost, prob in zip(grid_costs, binned_dist):
        if total_prob + prob < alpha:
            total_prob += prob
            total_cost += prob * cost
        else:
            total_cost += (alpha - total_prob) * cost
            break
    cvar = total_cost / alpha
    return cvar

def simulate_circuit(grid_costs, binned_dist, angles):
    assert len(grid_costs) == len(binned_dist)
    d = len(binned_dist)

    assert len(angles) % 2 == 0
    num_layers = len(angles) // 2

    phi = np.sqrt(binned_dist)
    psi = copy.deepcopy(phi)
    for i in range(num_layers):
        a, b = angles[2*i], angles[2*i+1]
        s = 0.0
        for k in range(d):
            s += np.exp(-1j*a*grid_costs[k]) * psi[k] * phi[k]
        s *= (np.exp(-1j*b) - 1)
        new_psi = np.zeros((d), dtype=complex)
        for k in range(d):
            new_psi[k] = psi[k] * np.exp(-1j*grid_costs[k]*a) + phi[k] * s
        psi = copy.deepcopy(new_psi)
    new_binned_dist = np.abs(psi) ** 2
    return new_binned_dist

def optimize_angles(
    grid_costs,
    binned_dist,
    num_layers,
    alpha,
    costs_for_cvar,
    ranks_for_cvar,
    opt_method="L-BFGS-B",
    maxiter=1000,
    num_trials=10
):
    assert len(grid_costs) == len(binned_dist)
    d = len(binned_dist)

    cost_function = lambda angles: compute_binned_cvar(
        grid_costs,
        simulate_circuit(grid_costs, binned_dist, angles),
        alpha
    )

    best_angles = None
    best_value = None

    for _ in range(num_trials):
        initial_angles = np.random.uniform(0.0, np.pi, 2*num_layers)

        result = scipy.optimize.minimize(
            cost_function,
            initial_angles,
            method=opt_method,
            options={'maxiter': maxiter},
        )
        opt_value = result.fun
        opt_angles = result.x

        if best_value is None or best_value > opt_value:
            best_angles = opt_angles
            best_value = opt_value

    return best_angles, best_value
