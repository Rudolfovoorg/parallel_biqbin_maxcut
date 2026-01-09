import json
import numpy as np
from abc import ABC, abstractmethod
from glob import glob

from biqbin.biqbin_base import ProblemMaxCut, ProblemQubo, SolutionMaxCut, SolutionQubo
from biqbin.utils import from_sparse, convert_numpy_to_json_serializable


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
        offset = 0
        if 'offset' in qubo_data:
            offset = qubo_data['offset']
        return ProblemQubo(Q=qubo, offset=offset, problem_name=self.problem_name, is_minimization=True, optimize_input=self.optimize_input)



class ToFile(ABC):
    """Base abstract class for saving the solution to disk. All subclasses must implenent the 
    `write` method that takes a solution and saves it as file.
    """

    @abstractmethod
    def write(self, filename: str, overwrite: bool = False, with_metadata: bool = True) -> None:
        ...

    def get_output_path(self, out_file: str, overwrite: bool) -> str:
        """Get the proper output path in case it already exists and we do not wish to overwrite.
        Attaches _N where N is the number of the next free output file. Adds .json if not already in the 
        out_file's name.

        Args:
            out_file (str): output file path.
            overwrite (bool): if overwriting the outfile will not be changed.

        Returns:
            str: output file path
        """
        out_file = out_file[:-5] if out_file.endswith('.json') else out_file
        if overwrite:
            return out_file

        file_count = len(glob(f'{out_file}*.json'))
        if file_count > 0:
            out_file += f'_{file_count}'
        return out_file + '.json'


class MaxCutToJson(ToFile):
    """Helper class to save the SolutionMaxCut as a json file.
    """

    def __init__(self, solution: SolutionMaxCut) -> None:
        self.solution: SolutionMaxCut = solution

    def write(self, filename: str, overwrite: bool = False, with_metadata: bool = True) -> None:
        """Save the solution as JSON file.

        Args:
            solution (SolutionMaxCut): Solution class returned by Biqbin after solving the problem
            with_metadata (bool, optional): Add meta_data to output file. Defaults to True.
        """

        # Check if output filename exists if we are not overriding and replace with filename_N.json
        output_path = self.get_output_path(filename, overwrite)

        save_output = {
            'maxcut': self.solution.solution
        }
        if with_metadata:
            save_output['meta_data'] = self.solution.meta_data

        with open(output_path, 'w') as f:
            json.dump(save_output, f,
                      default=convert_numpy_to_json_serializable)


class QuboToJson(ToFile):
    """Helper class to save QUBO solution as json file.
    """

    def __init__(self, solution: SolutionQubo) -> None:
        self.solution: SolutionQubo = solution

    def write(self, filename: str, overwrite: bool = False, with_metadata: bool = True, with_maxcut_solution: bool = False) -> None:
        """Save qubo solution to a json file.

        Args:
            solution (SolutionQubo): Solution class returned by Biqbin after solving the problem
            with_metadata (bool, optional): Add meta_data to output. Defaults to True.
            with_maxcut_solution (bool, optional): Add MaxCut solution to save output. Defaults to False.
        """

        # Check if output filename exists if we are not overriding and replace with filename_N.json
        output_path = self.get_output_path(filename, overwrite)

        save_output = {
            'qubo': self.solution.solution
        }

        if with_maxcut_solution:
            save_output['maxcut'] = self.solution.solution_maxcut
        if with_metadata:
            save_output['meta_data'] = self.solution.meta_data

        with open(output_path, 'w') as f:
            json.dump(save_output, f,
                      default=convert_numpy_to_json_serializable)
