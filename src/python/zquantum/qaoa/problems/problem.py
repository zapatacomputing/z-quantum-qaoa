from abc import ABC, abstractmethod
import networkx as nx
from openfermion import QubitOperator
from typing import Tuple, List
from ._problem_evaluation import (
    evaluate_solution,
    solve_graph_problem_by_exhaustive_search,
)


class Problem(ABC):
    @staticmethod
    @abstractmethod
    def build_hamiltonian(graph: nx.Graph):
        raise NotImplementedError

    @classmethod
    def get_hamiltonian(
        cls, graph: nx.Graph, scale_factor: float = 1.0, offset: float = 0.0, **kwargs
    ) -> QubitOperator:
        # Relabeling for monotonicity purposes
        num_nodes = range(len(graph.nodes))
        mapping = {node: new_label for node, new_label in zip(graph.nodes, num_nodes)}
        graph = nx.relabel_nodes(graph, mapping=mapping)

        hamiltonian = cls.build_hamiltonian(graph)

        hamiltonian.compress()

        return hamiltonian * scale_factor + offset

    @classmethod
    def evaluate_solution(cls, solution: Tuple[int], graph: nx.Graph) -> float:
        return evaluate_solution(solution, graph, cls.get_hamiltonian)

    @classmethod
    def solve_by_exhaustive_search(
        cls,
        graph: nx.Graph,
    ) -> Tuple[float, List[Tuple[int]]]:
        return solve_graph_problem_by_exhaustive_search(
            graph, cost_function=cls.evaluate_solution
        )
