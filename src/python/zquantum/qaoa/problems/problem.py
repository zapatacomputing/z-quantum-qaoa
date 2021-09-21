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
    def get_hamiltonian(
        graph: nx.Graph, scale_factor: float = 1.0, offset: float = 0.0, **kwargs
    ) -> QubitOperator:
        pass

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
