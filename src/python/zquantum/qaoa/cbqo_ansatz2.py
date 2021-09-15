from zquantum.core.interfaces.ansatz import Ansatz, ansatz_property
from zquantum.core import circuits

from typing import List, Optional, Union

import numpy as np
import sympy
from overrides import overrides
import copy

class CBQOAnsatz2(Ansatz):

    supports_parametrized_circuits = True

    def __init__(
        self,
        number_of_layers: int,
        costs: np.ndarray,
        seed_solution: np.ndarray,
        value_differences: np.ndarray,
        random_seed: int,
        use_complete_graph: Optional[bool] = True,
        number_of_segments: Optional[int] = 1,
        mixer_type: str = "Grover"
    ):
        super().__init__(number_of_layers)
        assert len(costs) == 2 ** len(seed_solution)
        self._number_of_qubits = len(seed_solution)
        self.number_of_layers = number_of_layers
        self.costs = costs
        self.seed_solution = seed_solution

        assert value_differences.shape[0] == self._number_of_qubits
        assert value_differences.shape[1] == self._number_of_qubits
        self.value_differences = value_differences

        self.number_of_segments = number_of_segments
        if use_complete_graph or np.sum(seed_solution) != self._number_of_qubits // 2:
            self.pair_sequence = generate_random_pair_sequence(self._number_of_qubits, seed=random_seed)
        else:
            self.pair_sequence = generate_pair_sequence(seed_solution)
        # print(self.pair_sequence)
        self.mixer_type = mixer_type

    @property
    def number_of_qubits(self) -> int:
        """Returns number of qubits used for the ansatz circuit.
        """
        return self._number_of_qubits

    @property
    def number_of_params(self) -> int:
        """Returns number of parameters in the ansatz.
        """
        return 2 * self.number_of_layers + 2

    @property
    def parametrized_circuit(self) -> circuits.Circuit:
        """Returns a parametrized circuit if given ansatz supports it."""
        if self._parametrized_circuit is None:
            if self.supports_parametrized_circuits:
                self._parametrized_circuit = self._generate_circuit()
            else:
                raise (
                    NotImplementedError(
                        "{0} does not support parametrized circuits.".format(
                            type(self).__name__
                        )
                    )
                )
        return self._parametrized_circuit

    @overrides
    def _generate_circuit(self, params: Optional[np.ndarray] = None) -> circuits.Circuit:
        """Returns a parametrizable circuit represention of the ansatz.
        Args:
            params: parameters of the circuit.
        """
        n = self.number_of_qubits

        if params is not None:
            assert len(params) == 2 * self.number_of_layers + 2
            gamma = params[0]
            theta = params[1]
            if self.number_of_layers > 0:
                angles = params[2:]
        else:
            gamma = sympy.Symbol(f"gamma")
            theta = sympy.Symbol(f"theta")
            angles = []
            for k in range(self.number_of_layers):
                angles.append(sympy.Symbol(f"alpha_{k}"))
                angles.append(sympy.Symbol(f"beta_{k}"))

        gates = []

        gates += [circuits.X(i) for i in range(n) if self.seed_solution[i]==1]

        gates += build_xy_mixing_circuit(
            self.number_of_qubits,
            self.pair_sequence,
            self.value_differences,
            gamma,
            theta,
            number_of_segments=self.number_of_segments
            )

        for k in range(self.number_of_layers):
            phases = [-angles[2*k]*cost for cost in self.costs]
            gates.append(circuits.MultiPhaseOperation(phases))

            gates += build_xy_mixing_circuit(
                self.number_of_qubits,
                self.pair_sequence[::-1],
                self.value_differences,
                -gamma,
                theta,
                number_of_segments=self.number_of_segments
                )

            gates += [circuits.X(i) for i in range(n) if self.seed_solution[i]==1]
            if self.mixer_type == "Grover":
                phases = [-angles[2*k+1]] + [0.0] * (2**n-1)
                gates.append(circuits.MultiPhaseOperation(phases))
            else:
                gates += [circuits.RZ(angles[2*k+1])(i) for i in range(n)]
            gates += [circuits.X(i) for i in range(n) if self.seed_solution[i]==1]

            gates += build_xy_mixing_circuit(
                self.number_of_qubits,
                self.pair_sequence,
                self.value_differences,
                gamma,
                theta,
                number_of_segments=self.number_of_segments
                )

        return circuits.Circuit(gates)


def generate_pair_sequence(seed_solution):
    n = len(seed_solution)
    assert n % 2 == 0
    m = n // 2

    A = []
    B = []
    for i in range(n):
        if seed_solution[i] == 1:
            A.append(i)
        else:
            B.append(i)
    assert len(A) == len(B)

    pairs = []
    for g in range(m):
        for i in range(m):
            pairs.append((A[i], B[(i+g) % m]))
    return pairs

def generate_random_pair_sequence(num_qubits, seed=None):
    if seed is not None:
        np.random.seed(seed)

    n = num_qubits
    m = n * (n-1) // 2

    original_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    permutated_pairs = np.random.permutation(original_pairs)

    idx = 0
    while idx < m:
        current_qubits = set()
        for idx2 in range(idx, m):
            a, b = permutated_pairs[idx2]
            if a not in current_qubits and b not in current_qubits:
                current_qubits.add(a)
                current_qubits.add(b)
                if idx2 != idx:
                    temp = copy.deepcopy(permutated_pairs[idx])
                    permutated_pairs[idx] = permutated_pairs[idx2]
                    permutated_pairs[idx2] = temp
                idx += 1
                if len(current_qubits) == n:
                    break

    return permutated_pairs

def build_xy_mixing_circuit(num_qubits, pairs, value_differences, gamma, theta, number_of_segments=1):
    gates = []
    for _ in range(number_of_segments):
        for a, b in pairs:
            assert a != b
            assert value_differences[a][b] != None
            # print(value_differences[a][b], gamma, theta)
            angle = gamma / (1.0 + np.exp(-theta * value_differences[a][b]))
            angle /= number_of_segments
            gates.append(circuits.XY(angle)(a, b))
    return gates
