import networkx as nx
from openfermion import QubitOperator
from .problem import Problem


class MaxCut(Problem):
    def _build_hamiltonian(self, graph: nx.Graph) -> QubitOperator:
        """Converts a MAXCUT instance, as described by a weighted graph, to an Ising
        Hamiltonian. It allows for different convention in the choice of the
        Hamiltonian. The returned Hamiltonian is consistent with the definitions from
        "A Quantum Approximate Optimization Algorithm" by E. Farhi, eq. 12
        (https://arxiv.org/pdf/1411.4028.pdf)
        and
        "Performance of the Quantum Approximate Optimization Algorithm on the Maximum
        Cut Problem" eq. 1 (https://arxiv.org/pdf/1811.08419.pdf).

        Note: In the convention we assumed, values of the cuts are negative
        to frame the problem as a minimization problem.
        So for a linear graph 0--1--2 with weights all equal 1, and the solution
        [0,1,0], the returned value will be equal to -2.

        Args:
            graph: undirected weighted graph defining the problem
            scale_factor: constant by which all the coefficients in the Hamiltonian
                will be multiplied
            offset: coefficient of the constant term added to the Hamiltonian to
                shift its energy levels

        Returns:
            operator describing the Hamiltonian
        """
        hamiltonian = QubitOperator()
        shift = 0.0

        for i, j in graph.edges:
            try:
                weight = graph.adj[i][j]["weight"]
            except KeyError:
                weight = 1

            hamiltonian += weight * QubitOperator(f"Z{i} Z{j}")
            shift -= weight

        return 0.5 * (hamiltonian + shift)
