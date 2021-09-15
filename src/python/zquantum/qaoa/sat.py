import numpy as np
import copy
from functools import reduce
from .utils import get_x_vec, NpEncoder
from openfermion import IsingOperator
import json

class MAXkSAT(object):
    def __init__(self, clauses, weights):
        self.clauses = clauses
        self.weights = weights
        self.num_clauses = len(clauses)

        self.literals = sorted(list(reduce(lambda a, b: a|b, clauses, set())))
        self.variables = sorted(list(set([literal//2 for literal in self.literals])))
        self.num_variables = len(self.variables)
        self.idx_to_var = {i:v for i, v in enumerate(self.variables)}
        self.var_to_idx = {v:i for i, v in self.idx_to_var.items()}

        if self.num_clauses > 0:
            self.clause_lens = [len(clause) for clause in clauses]
            self.max_clause_length = max(self.clause_lens)
            self.equal_clause_length = all(x==self.max_clause_length for x in self.clause_lens)

    def __str__(self):
        output = ""
        for clause, weight in zip(self.clauses, self.weights):
            output += "{:.2f} [".format(weight)
            cnt = 0
            for literal in clause:
                if literal % 2 == 1:
                    output += "not "
                output += "x{}".format(literal // 2)
                if cnt < len(clause)-1:
                    output += ", "
                cnt += 1
            output += "], "
        return output

    def evaluate_assignment(self, assignment):
        assert len(assignment) == len(self.variables)
        value = 0
        for clause, weight in zip(self.clauses, self.weights):
            for literal in clause:
                variable = literal // 2
                i = self.var_to_idx[variable]
                if (literal + 1) % 2 == assignment[i]:
                    value += weight
                    break
        return value

    def evaluate_all_possible_assignments(self):
        # Slow for large instances!
        values = []
        for i in range(2**self.num_variables):
            assignment = get_x_vec(i, n=self.num_variables)
            values.append(self.evaluate_assignment(assignment))
        return values

    def get_top_assignments(self, ratio=0.95):
        # Slow for large instances!
        assert self.num_clauses > 0
        values = self.evaluate_all_possible_assignments()
        max_value = max(values)
        min_value = min(values)
        mean_value = np.mean(values)
        # threshold = max_value * ratio
        threshold = mean_value + (max_value - mean_value) * ratio
        top_assignments_and_values = []
        for i, value in enumerate(values):
            if value >= threshold:
                top_assignments_and_values.append((get_x_vec(i, self.num_variables), value))
        return top_assignments_and_values, max_value, min_value, mean_value


def generate_random_k_sat_instance(
    num_variables,
    num_clauses,
    max_clause_length,
    equal_clause_length=False,
    allow_duplicates=False,
    weight_type="uniform"
):
    if equal_clause_length:
        literals = list(range(2*num_variables))
    else:
        literals = list(range(2*num_variables+1))

    clauses = []
    weights = []

    while len(clauses) < num_clauses:

        if equal_clause_length:
            candidate = set(np.random.choice(literals, max_clause_length, replace=False))
        else:
            candidate = set(np.random.choice(literals, max_clause_length, replace=True))
            candidate -= set([2*num_variables])

        if len(candidate) == 0:
            continue

        if any(set([2*i, 2*i+1]) <= candidate for i in range(num_variables)):
            continue

        if not allow_duplicates and any(candidate == clause for clause in clauses):
            continue

        candidate = set([int(x) for x in candidate])
        clauses.append(candidate)

        if weight_type == "uniform":
            weight = np.random.uniform(0.0, 1.0)
        else:
            weight = 1.0

        weights.append(weight)

    return MAXkSAT(clauses, weights)


def construct_ising_hamiltonian_for_max_k_sat(k_sat):
    output = IsingOperator()
    for clause, weight in zip(k_sat.clauses, k_sat.weights):
        temp = IsingOperator("", 1.0)
        for literal in clause:
            i = literal // 2
            if literal % 2 == 0: # x_i in clause
                temp *= IsingOperator("", 0.5) + IsingOperator("Z"+str(i), 0.5)
            else: # not x_i in clause
                temp *= IsingOperator("", 0.5) - IsingOperator("Z"+str(i), 0.5)
        term = (temp - IsingOperator("", 1.0)) * weight
        output += term
    return output

def save_k_sat_instance(k_sat, filename):
    data = {
        "clauses": [list(clause) for clause in k_sat.clauses],
        "weights": k_sat.weights
    }
    with open(filename, 'w') as outfile:
        json.dump(data, outfile, cls=NpEncoder)

def load_k_sat_instance(filename):
    with open(filename, 'r') as infile:
        data = json.load(infile)
    clauses = [set(clause) for clause in data["clauses"]]
    weights = data["weights"]
    return MAXkSAT(clauses, weights)
