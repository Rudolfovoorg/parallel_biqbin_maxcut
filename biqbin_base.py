__version__ = '2.0.0'

import numpy.typing as npt

from typing import Generic, TypeVar
from abc import ABC, abstractmethod
from glob import glob
import argparse
import numpy as np
import json
import pyqplib
from pyqplib.obj import QuadraticObjective

from utils import from_sparse, check_matrix_validity_wrap, convert_numpy_to_json_serializable, divide_matrix_by_gcd
from biqbin import (run, set_heuristic,
                    default_heuristic,
                    get_rank, set_read_data)


class PrettyPrint:
    def __str__(self) -> str:
        return (f'class: {type(self).__name__}')

    def __repr__(self) -> str:
        return str(self)


TProblem = TypeVar("TProblem", bound="ProblemMaxCut")
TSolution = TypeVar('TSolution', bound="SolutionMaxCut")


class ProblemMaxCut(Generic[TProblem], PrettyPrint):
    def __init__(self,
                 maxcut_adjacency_matrix: npt.NDArray[np.floating | np.integer],
                 problem_name: str,
                 optimize_input: bool = False) -> None:
        self._maxcut_adjacency_matrix = maxcut_adjacency_matrix
        self.problem_name = problem_name
        self.gcd: int | None = None
        self.optimize_mc_adj_matrix = optimize_input

    @property
    @check_matrix_validity_wrap
    def maxcut_adjacency_matrix(self):
        return self._maxcut_adjacency_matrix

    @maxcut_adjacency_matrix.setter
    def maxcut_adjacency_matrix(self, value: npt.NDArray[np.floating | np.integer]):
        if self.optimize_mc_adj_matrix:
            self.gcd, self._maxcut_adjacency_matrix = divide_matrix_by_gcd(
                value)
        else:
            self._maxcut_adjacency_matrix = value

    def __str__(self) -> str:
        return f'{super().__str__()}\nMaxCut adjacency matrix =\n{self.maxcut_adjacency_matrix}'


class ProblemQubo(ProblemMaxCut):
    def __init__(self, Q: np.ndarray, problem_name: str,  is_minimization: bool, optimize_input: bool):
        self.Q: np.ndarray = Q
        self.is_minimization: bool = is_minimization
        if self.is_minimization:
            super().__init__(self.qubo2maxcut(Q), problem_name, optimize_input)
        else:
            super().__init__(self.qubo2maxcut(-Q), problem_name, optimize_input)

    @check_matrix_validity_wrap
    def qubo2maxcut(self, qubo: np.ndarray) -> np.ndarray:
        """Convert qubo to adjacency matrix that biqbin can read, 
        optionally optimizes the input data by dividing the values by the gcd.

        Args:
            qubo (np.ndarray): qubo as 2d numpy array
        Returns:
            np.ndarray: adjacency matrix for max cut problem
        """

        q_sym = 1/2*(qubo.T + qubo)

        q_int = np.array(q_sym, dtype=np.int64)
        if not np.all(q_sym == q_int):
            raise ValueError("All QUBO values need to be integers!")

        Qe_plus_c = -np.array([(np.sum(q_sym, 1))])
        np.fill_diagonal(q_sym, 0)

        return np.block([
            [q_sym, Qe_plus_c.T],
            [Qe_plus_c, np.zeros((1, 1))]
        ])

    def __str__(self) -> str:
        return f'{super().__str__()}\nQubo =\n{self.Q}'


class SolutionMaxCut(Generic[TProblem], PrettyPrint):
    def __init__(self, biqbin_result: dict, problem: TProblem) -> None:
        if problem.optimize_mc_adj_matrix and problem.gcd != 1:
            biqbin_result['maxcut']['computed_val'] *= problem.gcd

        self.__solution: dict = biqbin_result['maxcut']
        self.meta_data: dict = biqbin_result['meta_data']
        self.problem: TProblem = problem

    @property
    def solution(self):
        return self.__solution

    def __str__(self) -> str:
        return (f'{super().__str__()}\n'
                f'--- Max-Cut ---\n'
                f' Computed value = {self.__solution['computed_val']}\n'
                f'Solution MaxCut = {self.__solution['solution']}\n'
                f'              x = {self.__solution['x']}\n')


class SolutionQubo(SolutionMaxCut[ProblemQubo]):
    def __init__(self, biqbin_result: dict, problem: ProblemQubo) -> None:
        super().__init__(biqbin_result, problem)
        self.solution_maxcut = super().solution
        self.__solution = self.maxcut2qubo()

    @property
    def solution(self):
        return self.__solution

    def maxcut2qubo(self) -> dict:
        """Write qubo solution from the retrievied maxcut solution

        Returns:
            dict: Qubo specific result (computed_val, x, solution, cardinality, obj)
        """
        qubo_solution, qubo_x = self._maxcut_solution2qubo_solution(
            self.solution_maxcut["solution"]
        )

        computed_val = float(self.problem.Q.dot(qubo_x).dot(qubo_x))
        return {'computed_val': computed_val,
                'solution': qubo_solution,
                'x': qubo_x,
                'cardinality': float(sum(qubo_x)),
                'minimization': self.problem.is_minimization
                }

    def _maxcut_solution2qubo_solution(self, maxcut_solution: np.ndarray):
        """Convert maxcut solution nodes to qubo solution nodes

        Args:
            maxcut_solution (np.ndarray): maxcut solution found by biqbin
        Returns:
            np.ndarray: qubo solution nodes
        """

        n, _ = self.problem.Q.shape

        _x_mc = np.array(maxcut_solution, dtype=int)-1
        x_mc_sol = np.ones(n + 1)
        xx = np.zeros(n + 1, dtype=int)
        xx[_x_mc] = 1

        x_mc_sol[_x_mc] = -1
        x_mc_sol *= -x_mc_sol[-1]
        y = 1/2*(x_mc_sol+1)[:-1]
        qubo_solution = np.nonzero(y)[0] + 1

        return qubo_solution.tolist(), y.astype(int).tolist()

    def __str__(self):
        return (
            f'{super().__str__()}\n'
            f'--- QUBO ---\n'
            f' Computed value = {self.__solution['computed_val']}\n'
            f'       Solution = {self.__solution['solution']}\n'
            f'              x = {self.__solution['x']}\n'
            f'is_minimization = {self.problem.is_minimization}\n'
        )


class LoadFromFile(ABC):
    """Interface for all file loader classes, subclasses must implement a `read` method,
    that takes a filename and returns a ProblemMaxCut or it's subclass (i.e. ProblemQubo) 
    """
    @abstractmethod
    def read(self, filename: str, problem_name: str | None = None, optimize_input: bool = False) -> ProblemMaxCut:
        ...


class SaveToFile(ABC, Generic[TSolution]):
    """Base abstract class for saving the solution to disk. All subclasses must implenent the 
    `write` method that takes a solution and saves it as file.
    """
    @abstractmethod
    def write(self, solution: TSolution, filename: str, overwrite: bool = False, with_metadata: bool = True) -> None:
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


class MaxCutFromJson(LoadFromFile):
    def read(self, filename: str, problem_name: str | None = None, optimize_input: bool = False) -> ProblemMaxCut:
        """Read MaxCut json file that contains the adjacency matrix in sparse format.

        Args:
            filename (str): Path to json file containing 'maxcut' key and sparse adjacency matrix as value.
            problem_name (str | None, optional): Name of the problem instance if it is different than filename. Defaults to None.
            optimize_input (bool, optional): Divide the biqbin input matrix by its greatest common divisor. Defaults to False.

        Returns:
            ProblemMaxCut: _description_
        """
        with open(filename, "r") as f:
            mc_data = json.load(f)

        if problem_name is None:
            problem_name = filename
        adj_matrix = from_sparse(mc_data["maxcut"])
        return ProblemMaxCut(adj_matrix, problem_name, optimize_input=optimize_input)


class MaxCutFromEdgeWeights(LoadFromFile):
    def read(self, filename: str, problem_name: str | None = None, optimize_input: bool = False) -> ProblemMaxCut:
        """Read MaxCut edge weight file and return the MaxCutProblem.

        Args:
            filename (str): path to edge weight file. 
            problem_name (str | None, optional): Name of the problem instance if it is different than filename. Defaults to None.
            optimize_input (bool, optional): Divide the biqbin input matrix by its greatest common divisor. Defaults to False.

        Returns:
            ProblemMaxCut: Problem that can be passed into MaxCutSolver.compute.
        """
        with open(filename, 'r') as f:
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
        if problem_name is None:
            problem_name = filename
        return ProblemMaxCut(adj_matrix, problem_name, optimize_input=optimize_input)


class MaxCutToJson(SaveToFile):
    """Helper class to save the SolutionMaxCut as a json file.
    """
    def write(self, solution: SolutionMaxCut, filename: str, overwrite: bool = False, with_metadata: bool = True) -> None:
        """Save the solution as JSON file.

        Args:
            solution (SolutionMaxCut): Solution class returned by Biqbin after solving the problem
            with_metadata (bool, optional): Add meta_data to output file. Defaults to True.
        """

        # Check if output filename exists if we are not overriding and replace with filename_N.json
        output_path = self.get_output_path(filename, overwrite)

        save_output = {
            'maxcut': solution.solution
        }
        if with_metadata:
            save_output['meta_data'] = solution.meta_data

        with open(output_path, 'w') as f:
            json.dump(save_output, f,
                      default=convert_numpy_to_json_serializable)


class QuboToJson(SaveToFile):
    """Helper class to save QUBO solution as json file.
    """

    def write(self, solution: SolutionQubo, filename: str, overwrite: bool = False, with_metadata: bool = True, with_maxcut_solution: bool = False) -> None:
        """Save qubo solution to a json file.

        Args:
            solution (SolutionQubo): Solution class returned by Biqbin after solving the problem
            with_metadata (bool, optional): Add meta_data to output. Defaults to True.
            with_maxcut_solution (bool, optional): Add MaxCut solution to save output. Defaults to False.
        """

        # Check if output filename exists if we are not overriding and replace with filename_N.json
        output_path = self.get_output_path(filename, overwrite)

        save_output = {
            'qubo': solution.solution
        }

        if with_maxcut_solution:
            save_output['maxcut'] = solution.solution_maxcut
        if with_metadata:
            save_output['meta_data'] = solution.meta_data

        with open(output_path, 'w') as f:
            json.dump(save_output, f,
                      default=convert_numpy_to_json_serializable)


class QuboFromJson(LoadFromFile):
    """Reads qubo instance file, should be a json dictionary with "qubo" key
    and a COO sparse matrix with data, row and col.
    """

    def read(self, filename: str, problem_name: str | None = None, optimize_input: bool = False) -> ProblemQubo:
        """Read from the given filename and return the ProblemQubo used by the QuboSolver.

        Args:
            filename (str): path to json file containing 'qubo' key and sparse matrix presentation as value. 
            problem_name (str | None, optional): Name of the problem instance if it is different than filename. Defaults to None.
            optimize_input (bool, optional): Divide the biqbin input matrix by its greatest common divisor. Defaults to False.

        Returns:
            ProblemQubo: Qubo Problem class that can be passed into QuboSolver.compute method.
        """
        with open(filename, "r") as f:
            qubo_data = json.load(f)

        if problem_name is None:
            problem_name = filename
        qubo = from_sparse(qubo_data["qubo"])
        return ProblemQubo(Q=qubo, problem_name=problem_name, is_minimization=True, optimize_input=optimize_input)


class QuboFromQPLIB(LoadFromFile):
    """DataGetter for QPLIB instances https://qplib.zib.de/, 
    only unconstrained binary problems are allowed.
    """

    def read(self, filename: str, problem_name: str | None = None, optimize_input: bool = False) -> ProblemQubo:
        """Reads .qplib format and constructs a qubo. 
        Checks if the problem itself is valid for Biqbin solver, while the integer check 
        is done when converting the constructed QUBO to Max-Cut form.

        Args:
            filename (str): path to qplib file. 
            problem_name (str | None, optional): Name of the problem instance if it is different than filename. Defaults to None.
            optimize_input (bool, optional): Divide the biqbin input matrix by its greatest common divisor. Defaults to False.

        Raises:
            ValueError: Only unconstrained problems are valid
            ValueError: Only binary problems are valid
            ValueError: Only quadratic problems are valid

        Returns:
            ProblemQUBO
        """
        qplib_problem = pyqplib.read_problem(filename)
        # Check if reading qplib format worked
        if not isinstance(qplib_problem, pyqplib.Problem):
            raise ValueError(
                f"Failed reading qplib problem at path {filename}!")

        if problem_name is None:
            problem_name = filename

        # Check if the problem fits the solver
        if qplib_problem.description.cons_type != pyqplib.ProblemConsType.UNCONSTRAINED:
            raise ValueError("Biqbin can only handle unconstrained problems!")
        if qplib_problem.description.var_type != pyqplib.ProblemVarType.BINARY:
            raise ValueError("Problem is not binary!")
        if not isinstance(qplib_problem.obj, QuadraticObjective):
            raise ValueError("Problem is not quadratic!")

        # pyqplib has it's own matrix representation
        qubo = qplib_problem.obj.mat.full().todense().T

        qubo = np.triu(qubo) / 2
        qubo += np.diag(qplib_problem.obj.lin)

        # Update goal of the objective function
        minimize = qplib_problem.obj.sense == pyqplib.Sense.MINIMIZE

        return ProblemQubo(Q=qubo, problem_name=problem_name, is_minimization=minimize, optimize_input=optimize_input)


class MaxCutSolver(Generic[TProblem], PrettyPrint):
    """Default MaxCut Biqbin Python Wrapper, runs Biqbin MaxCut using its original C-functions
    """
    solver_name = f'PyBiqBin-MaxCut {__version__}'

    def __init__(self, params: str, time_limit: int = 0):
        """Initialize the solver

        Args:
            params (str): path to parameters file
            time_limit (int): time limit in seconds
        """
        self.problem: TProblem | None = None
        self.params = params
        self.time_limit = time_limit
        set_read_data(self.read_data)
        set_heuristic(self.heuristic)

    @check_matrix_validity_wrap
    def read_data(self) -> np.ndarray:
        """Return the maxcut_adjacency_matrix from Problem

        Returns:
            np.ndarray: adjacency matrix
        """
        if self.problem is None:
            raise ValueError("Problem instance not set!")

        return self.problem.maxcut_adjacency_matrix

    def heuristic(self, L0: np.ndarray, L: np.ndarray, xfixed: np.ndarray, sol_X: np.ndarray, x: np.ndarray) -> float:
        """Default GW heuristic (heuristic_unpacked in heuristic.c)

        Args:
            L0 (np.ndarray): original Problem *SP->L matrix
            L (np.ndarray): subproblem Problem *PP->L matrix
            xfixed (np.array): current branch and bound node fixed variable array
            sol_X (np.array): current solution stored in the branch and bound node
            x (np.array): stores the solution of the heuristic function, used by the solver to determine the lower bound

        Returns:
            float: value of the solution array "x" found by the heuristic function
        """
        return default_heuristic(L0, L, xfixed, sol_X, x)

    def _run_solver(self) -> dict | None:
        """Runs Biqbin C/C++ implementation

        Raises:
            ValueError: If no result is retrieved on the master process.

        Returns:
            dict | None: Solution python dict built by C++, or None on MPI rank != 0
        """
        if self.problem is None:
            raise ValueError("Problem instance not set!")

        biqbin_result = run(self.solver_name, self.problem.problem_name,
                            self.params, self.time_limit)

        if (self.get_rank() == 0):
            if biqbin_result is None:
                raise ValueError(
                    'Result from BiqBin is None, computation failed!')

            biqbin_result['meta_data']['instance'] = self.problem.problem_name
            biqbin_result['meta_data']['parameters'] = {
                'time_limit': self.time_limit if self.time_limit > 0 else None,
                'optimized': self.problem.optimize_mc_adj_matrix,
                'gcd': self.problem.gcd
            }
            return biqbin_result

    def compute(self, problem: TProblem) -> SolutionMaxCut | None:
        """Compute the MaxCut solution using Biqbin

        Args:
            problem (ProblemMaxCut): problem to be solved.

        Returns:
            SolutionMaxCut | None: Returns the solution class on MPI rank == 0.
        """
        self.problem = problem

        biqbin_result = self._run_solver()
        if biqbin_result is not None:
            return SolutionMaxCut(biqbin_result, self.problem)
        else:
            return None

    def get_rank(self) -> int:
        """MPI process rank

        Returns:
            int: rank
        """
        return get_rank()


class QUBOSolver(MaxCutSolver[ProblemQubo]):
    solver_name = f'PyBiqBin-QUBO {__version__}'

    def compute(self, problem: ProblemQubo) -> SolutionQubo | None:
        """Computes the solution to the QUBO using Biqbin, only MPI rank == 0 returns Solution

        Returns:
            SolutionQubo: Solution class for QUBO problem. returns None if MPI rank != 0.
        """
        self.problem = problem

        biqbin_result = self._run_solver()

        if biqbin_result is not None:
            return SolutionQubo(biqbin_result, self.problem)
        else:
            return None


class BaseParser(argparse.ArgumentParser):
    def __init__(self, prog: str, description: str):
        super().__init__(prog=prog, description=description,
                         usage=f'mpirun [-n N] python3 {prog} problem_instance [-p PARAMS] [-w] [-o OUTPUT]',
                         epilog='For more information please visit https://github.com/Rudolfovoorg/parallel_biqbin_maxcut',
                         )
        self.add_argument('problem_instance',
                          help='Path to the problem instance file')

        # Optional arguments
        self.add_argument('-p', '--params', default='params',
                          help='custom parameters file path (default: "params")')
        self.add_argument('-w', '--overwrite',
                          action='store_true',
                          help='overwrite output.json instead of labeling with _NUMBER'
                          )
        self.add_argument('-O', '--optimize', action='store_true',
                          help='Divides the final input matrix values by their GCD')
        self.add_argument('-o', '--output', help='set custom output file path')
        # time limit format taken from SLURM docs https://slurm.schedmd.com/sbatch.html
        self.add_argument('-t', '--time', default='0', type=self.parse_time_limit,
                          help='set running time limit; acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"')

    def parse_time_limit(self, s: str) -> int:
        """
        Parse Slurm-style time limits:
        - "MM" (minutes only)
        - "HH:MM:SS"
        - "D-HH:MM:SS"
        Returns:
            int: total seconds
        """
        # If format includes days
        if "-" in s:
            days_str, rest = s.split("-", 1)
            days = int(days_str)
        else:
            days, rest = 0, s

        parts = rest.split(":")
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
        elif len(parts) == 2:
            hours, minutes = map(int, parts)
            seconds = 0
        elif len(parts) == 1:
            # Slurm allows just minutes like "30"
            return int(parts[0]) * 60
        else:
            raise argparse.ArgumentTypeError(f"Invalid time format: {s}")

        total_seconds = days*86400 + hours*3600 + minutes*60 + seconds
        return int(total_seconds)


class ParserMaxCut(BaseParser):
    def __init__(self):
        super().__init__(prog=f'biqbin_maxcut.py', description='Biqbin Maxcut solver')
        self.add_argument('-e', '--edge_weight',
                          action='store_true', help='use edge weight input file')


class ParserQubo(BaseParser):
    def __init__(self, prog=f'biqbin_qubo.py', description='Biqbin QUBO solver'):
        super().__init__(prog=prog, description=description)
        self.add_argument('--qplib', action='store_true',
                          help='Use .qplib file format')


class ParserDWaveHeuristic(ParserQubo):
    def __init__(self):
        super().__init__(prog='biqbin_heuristic.py',
                         description='Biqbin QUBO solver with DWave heuristic')
        self.add_argument('-d', '--debug', action='store_true',
                          help='enable debug logs')
        self.add_argument('-i', '--info', action='store_true',
                          help='enable info logs')
