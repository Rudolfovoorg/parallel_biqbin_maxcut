import json
import numpy as np
from abc import ABC, abstractmethod

from biqbin_base import ProblemMaxCut, ProblemQubo
from utils import from_sparse


class FromFile(ABC):
    """Interface for all file loader classes, subclasses must implement a `read` method,
    that takes a filename and returns a ProblemMaxCut or it's subclass (i.e. ProblemQubo) 
    """

    def __init__(self, filename: str, problem_name: str | None = None, optimize_input: bool = False) -> None:
        self.filename = filename
        self.problem_name = problem_name if problem_name is not None else filename
        self.optimize_input = optimize_input

    @abstractmethod
    def read(self) -> ProblemMaxCut:
        ...


class MaxCutFromJson(FromFile):
    def read(self) -> ProblemMaxCut:
        """Read MaxCut json file that contains the adjacency matrix in sparse format.

        Args:
            filename (str): Path to json file containing 'maxcut' key and sparse adjacency matrix as value.
            problem_name (str | None, optional): Name of the problem instance if it is different than filename. Defaults to None.
            optimize_input (bool, optional): Divide the biqbin input matrix by its greatest common divisor. Defaults to False.

        Returns:
            ProblemMaxCut: _description_
        """
        with open(self.filename, "r") as f:
            mc_data = json.load(f)

        adj_matrix = from_sparse(mc_data["maxcut"])
        return ProblemMaxCut(adj_matrix, self.problem_name, optimize_mc_adj_matrix=self.optimize_input)


class MaxCutFromEdgeWeights(FromFile):
    def read(self) -> ProblemMaxCut:
        """Read MaxCut edge weight file and return the MaxCutProblem.

        Args:
            filename (str): path to edge weight file. 
            problem_name (str | None, optional): Name of the problem instance if it is different than filename. Defaults to None.
            optimize_input (bool, optional): Divide the biqbin input matrix by its greatest common divisor. Defaults to False.

        Returns:
            ProblemMaxCut: Problem that can be passed into MaxCutSolver.compute.
        """
        with open(self.filename, 'r') as f:
            # Read number of vertices and edges
            num_vertices, num_edges = map(int, f.readline().split())
            adj_matrix = np.zeros(
                (num_vertices, num_vertices), dtype=np.float64)

            for _ in range(num_edges):
                i, j, weight = f.readline().split()
                i, j = int(i) - 1, int(j) - 1  # Convert to zero-based indexing
                weight = float(weight)

                adj_matrix[i, j] = weight
                adj_matrix[j, i] = weight

        return ProblemMaxCut(adj_matrix, self.problem_name, optimize_mc_adj_matrix=self.optimize_input)


class QuboFromJson(FromFile):
    """Reads qubo instance file, should be a json dictionary with "qubo" key
    and a COO sparse matrix with data, row and col.
    """

    def read(self) -> ProblemQubo:
        """Read from the given filename and return the ProblemQubo used by the QuboSolver.

        Args:
            filename (str): path to json file containing 'qubo' key and sparse matrix presentation as value. 
            problem_name (str | None, optional): Name of the problem instance if it is different than filename. Defaults to None.
            optimize_input (bool, optional): Divide the biqbin input matrix by its greatest common divisor. Defaults to False.

        Returns:
            ProblemQubo: Qubo Problem class that can be passed into QuboSolver.compute method.
        """
        with open(self.filename, "r") as f:
            qubo_data = json.load(f)

        qubo = from_sparse(qubo_data["qubo"])
        return ProblemQubo(Q=qubo, problem_name=self.problem_name, is_minimization=True, optimize_input=self.optimize_input)
