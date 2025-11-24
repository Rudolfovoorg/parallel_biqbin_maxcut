__version__ = '2.0.0'

from abc import ABC, abstractmethod
import argparse
from functools import reduce
from math import gcd
import numpy as np
import scipy as sp
import json
import warnings
from glob import glob
import pyqplib

from biqbin import (run, set_heuristic,
                    default_heuristic,
                    get_rank, set_read_data,
                    default_read_data)


class DataGetter(ABC):
    """Base class for all data getters.

    Subclasses are responsible for reading a problem instance from some
    source (file, generator, API, …) and exposing it in a format that can
    be passed to a solver.

    Subclasses must implement:
        `problem_instance_name` -> str:
            Returns a human-readable identifier of the problem instance.
        `problem_instance` -> np.ndarray:
            Returns the problem instance as a numpy ndarray in the format
            expected by the solver.
        `read_file` -> np.ndarray:
            Reads the raw data (e.g. from disk) and initializes internal
            state used by ``problem_instance()``.
    """
    @abstractmethod
    def problem_instance_name(self) -> str:
        """Gets the identifier string of the problem instance.

        Returns:
            str: name, file path, identifier, ... of the problem instance
        """
        ...

    @abstractmethod
    def problem_instance(self) -> np.ndarray:
        """Return the problem instance matrix.

        Returns:
            np.ndarray: A 2D numpy array of shape (n, n)
        """
        ...

    @abstractmethod
    def read_file(self) -> np.ndarray:
        """Called when Biqbin is ran, reads the problem instance from file
        and converts it to a numpy ndarray.

        Returns:
            np.ndarray: A 2D numpy array of shape (n, n)
        """
        ...

    def from_sparse(self, sparse_matrix) -> np.ndarray:
        """Helper function that converts from sparse coo matrix to regular form

        Args:
            sparse_matrix (dict): scipy sparse coo matrix

        Returns:
            np.ndarray: regular form matrix
        """
        return sp.sparse.coo_matrix(
            (sparse_matrix['data'],
             (sparse_matrix['row'], sparse_matrix['col'])),
            shape=sparse_matrix['shape'], dtype='float'
        ).todense().getA()


class DataGetterMaxCutDefault(DataGetter):
    """
    Uses the default C implementation or MaxCut, reads and parses maxcut instance file in edge weight list format and parses
    into the adjacency matrix.
    """

    def __init__(self, filename: str):
        self.filename: str = filename
        self.adj_matrix: np.ndarray | None = None

    def problem_instance_name(self) -> str:
        """Get the instance file path

        Returns:
            str: path to instance file
        """
        return self.filename

    def problem_instance(self) -> np.ndarray:
        """Gets the adjacency matrix from the instance file
        """
        if isinstance(self.adj_matrix, np.ndarray):
            return self.adj_matrix
        else:
            raise ValueError(
                f"Expected self.adj_matrix to be of type 'numpy.ndarray', but got '{type(self.adj_matrix)}'.")

    def read_file(self) -> np.ndarray:
        self.adj_matrix = default_read_data(self.filename)
        if isinstance(self.adj_matrix, np.ndarray):
            return self.adj_matrix
        else:
            raise ValueError(
                f"Expected self.adj_matrix to be of type 'numpy.ndarray', but got '{type(self.adj_matrix)}'.")


class DataGetterAdjacencyJson(DataGetterMaxCutDefault):
    """
    DataGetter for the Maxcut class, reads and parses json serialized dict with adj key and sparse coo matrix as value.
    """

    def read_file(self) -> np.ndarray:
        with open(self.filename, "r") as f:
            mc_data = json.load(f)

        self.adj_matrix = self.from_sparse(mc_data["adjacency"])

        adj_int = np.array(self.adj_matrix, dtype=np.int64)
        if not np.all(self.adj_matrix == adj_int):
            raise ValueError(
                "All values in the adjacency matrix need to be integers!")

        return self.adj_matrix


class MaxCutSolver:
    """Default MaxCut Biqbin Wrapper, runs Biqbin MaxCut using its original functions
    """
    solver_name = f'PyBiqBin-MaxCut {__version__}'

    def __init__(self, data_getter: DataGetter, params: str, time_limit: int = 0):
        """Initialize the solver

        Args:
            problem_instance_name (str): path to problem instance in edge weight list format
            params (str): path to parameters file
        """
        self.data_getter: DataGetter = data_getter
        self.params = params
        set_read_data(self.read_data)
        set_heuristic(self.heuristic)
        self.time_limit = time_limit

    def read_data(self) -> np.ndarray:
        """Transform edge weight list into an adjacancy matrix

        Returns:
            np.ndarray: adjacency matrix
        """
        return self.data_getter.read_file()

    def heuristic(self, L0: np.ndarray, L: np.ndarray, xfixed: np.ndarray, sol_X: np.ndarray, x: np.ndarray) -> float:
        """Default heuristic (heuristic_unpacked in heuristic.c)

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

    def run(self):
        """Runs Biqbin Maxcut solver

        Returns:
            dict: result dict with keys: "max_val" - max cut solution value, "solution" - nodes in this solution, "time" - spent solving 
        """
        result = run(self.solver_name, self.data_getter.problem_instance_name(
        ), self.params, self.time_limit)
        if (self.get_rank() == 0):
            result['meta_data']['instance'] = self.data_getter.problem_instance_name()
            result['meta_data']['parameters'] = {
                'time_limit': self.time_limit if self.time_limit > 0 else None}
            return result
        else:
            return None

    def get_rank(self) -> int:
        """MPI process rank

        Returns:
            int: rank
        """
        return get_rank()

    def save_result(self, result, output_path_in=None, overwrite=False):
        """Save the result dictionary as JSON file.

        Args:
            result (dict): result dictionary returned by biqbin
            output_path (str, optional): custom path to an output file. Defaults to None.
        """
        if not output_path_in:
            output_path = self.data_getter.problem_instance_name() + '.output'
        else:
            output_path = output_path_in

        # Always overwrite if filepath specific
        if not overwrite and not output_path_in:
            file_count = len(glob(f'{output_path}*.json'))
            if file_count > 0:
                output_path += f'_{file_count}'
        if not output_path_in:
            output_path += '.json'
        with open(output_path, "w") as f:
            json.dump(result, f, default=self._convert_numpy)

    def _convert_numpy(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class DataGetterQUBO(DataGetter):
    """Abstract class for QUBO DataGetters from which all DataGetters for QUBOSolver must inherit.
    It is responsible for providing the solver with the qubo in upper triangular np.ndarray and 

    Subclasses must implement:
        `problem_instance_name` -> str:
            Returns a human-readable identifier of the problem instance.
        `problem_instance` -> np.ndarray:
            Returns the problem instance as a numpy ndarray in the format
            expected by the solver.
        `read_file` -> np.ndarray:
            Reads the raw data (e.g. from disk) and initializes internal
            state used by ``problem_instance()``.
        `minimize` -> bool:
            return True if the objective is to minimize, False if maximize.
    """

    @abstractmethod
    def minimize(self) -> bool:
        """Defines if the problem is to minimize or maximize the objective function.

        Returns:
            bool: True - minimize, False - maximize
        """
        ...

    def get_qubo4biqbin(self) -> np.ndarray:
        """Wrapper for read_file() method, calls `read_file` and 
        returns -qubo if the objective is to maximize.

        Returns:
            np.ndarray: qubo if minimize(), else -qubo
        """
        qubo = self.read_file()
        return qubo if self.minimize() else -qubo

    def get_qubo_result(self, result: dict) -> dict:
        """Add 'qubo' result, built from the 'maxcut' result dict biqbin returned

        Args:
            result (dict): Biqbin result dictionary containing maxcut results

        Returns:
            dict: Qubo specific result (computed_val, x, solution, cardinality, obj)
        """
        qubo_solution, qubo_x = self._maxcut_solution2qubo_solution(
            result["maxcut"]["solution"]
        )

        computed_val = float(self.problem_instance().dot(qubo_x).dot(qubo_x))
        return {'computed_val': computed_val,
                'solution': qubo_solution,
                'x': qubo_x,
                'cardinality': float(sum(qubo_x)),
                'obj': 'minimize' if self.minimize() else 'maximize'
                }

    def _maxcut_solution2qubo_solution(self, maxcut_solution: np.ndarray):
        """Convert maxcut solution nodes to qubo solution nodes

        Args:
            maxcut_solution (np.ndarray): maxcut solution found by biqbin
        Returns:
            np.ndarray: qubo solution nodes
        """

        n, _ = self.problem_instance().shape

        _x_mc = np.array(maxcut_solution, dtype=int)-1
        x_mc_sol = np.ones(n + 1)
        xx = np.zeros(n + 1, dtype=int)
        xx[_x_mc] = 1

        x_mc_sol[_x_mc] = -1
        x_mc_sol *= -x_mc_sol[-1]
        y = 1/2*(x_mc_sol+1)[:-1]
        qubo_solution = np.nonzero(y)[0] + 1

        return qubo_solution.tolist(), y.astype(int).tolist()


class DataGetterJson(DataGetterQUBO):
    """Reads qubo instance file, should be a json dictionary with "qubo" key
    and a COO sparse matrix with data, row and col. Indices starts from zero.
    """

    def __init__(self, filename: str):
        """Load data from json file and save the data and qubo

        Args:
            filename (str): path to file
        """
        self.filename: str = filename
        self.qubo: np.ndarray | None = None

    def problem_instance_name(self) -> str:
        """Get the instance file path

        Returns:
            str: path to instance file
        """
        return self.filename

    def problem_instance(self) -> np.ndarray:
        """Gets the qubo in upper triangular form

        Returns:
            nd.ndarray: qubo
        """
        if isinstance(self.qubo, np.ndarray):
            return self.qubo
        else:
            raise ValueError(
                f'Expected self.qubo to be of type numpy.ndarray, got {type(self.qubo)}.')

    def read_file(self) -> np.ndarray:
        with open(self.filename, "r") as f:
            self.qubo_data = json.load(f)

        self.qubo = self.from_sparse(self.qubo_data["qubo"])
        return self.qubo

    def minimize(self) -> bool:
        """DataGetterJson reads minimization problems by default

        Returns:
            bool: True - minimize
        """
        return True


class DataGetterQPLIB(DataGetterQUBO):
    """DataGetter for QPLIB instances https://qplib.zib.de/, 
    only unconstrained binary problems are allowed. 

    Args:
        DataGetterQUBO (class): inherits from DataGetterQUBO, the default qubo datagetter
    """

    def __init__(self, filename: str):
        self.filename: str = filename
        self._minimize: bool = True
        self.qubo: np.ndarray | None
        self.pqlib_problem: pyqplib.desc.ProblemDescription | pyqplib.Problem | None = None

    def read_file(self):
        """Reads .qplib format and constructs a qubo. 
        Checks if the problem itself is valid for Biqbin solver, while the integer check 
        is done when converting the constructed QUBO to Max-Cut form.

        Raises:
            ValueError: Only unconstrained problems are valid
            ValueError: Only binary problems are valid

        Returns:
            np.ndarray: constructed qubo in upper triangular form
        """
        self.pqlib_problem = pyqplib.read_problem(self.filename)

        # Check if the problem fits the solver
        if self.pqlib_problem.description.cons_type != pyqplib.ProblemConsType.UNCONSTRAINED:
            raise ValueError("Biqbin can only handle unconstrained problems!")
        if self.pqlib_problem.description.var_type != pyqplib.ProblemVarType.BINARY:
            raise ValueError("Problem is not binary!")

        # pyqplib has it's own matrix represantation
        self.qubo = self.pqlib_problem.obj.mat.full().todense().T

        self.qubo = np.triu(self.qubo) / 2
        self.qubo += np.diag(self.pqlib_problem.obj.lin)

        # Update goal of the objective function
        self._minimize = self.pqlib_problem.obj.sense == pyqplib.Sense.MINIMIZE
        return self.qubo

    def minimize(self) -> bool:
        return self._minimize

    def problem_instance(self) -> np.ndarray:
        if isinstance(self.qubo, np.ndarray):
            return self.qubo
        else:
            raise ValueError(f'Expected self.qubo to be of type numpy.ndarray, got {type(self.qubo)}.')

    def problem_instance_name(self) -> str:
        return self.filename


class QUBOSolver(MaxCutSolver):
    solver_name = f'PyBiqBin-QUBO {__version__}'

    def __init__(self, data_getter: DataGetterQUBO, params: str, optimize_input: bool = False, time_limit: int = 0):
        super().__init__(data_getter, params, time_limit)
        self.optimize_input: bool = optimize_input
        self.gcd: int = 1

    def read_data(self) -> np.ndarray:
        """Read qubo json file, return an adjacency matrix for maxcut

        Returns:
            np.ndarray: adjacency matrix
        """

        return self._qubo2maxcut(self.data_getter.get_qubo4biqbin())

    def run(self) -> dict | None:
        """Runs the original biqbin then adds the qubo solution nodes to the result dict

        Returns:
            dict: result dict containing "maxcut" and "qubo" keys with their respective solutions
        """
        result = super().run()
        if self.get_rank() == 0:
            if result is None:
                raise ValueError("result is None, solution not retrieved!")

            result['maxcut']['computed_val'] *= self.gcd
            result['meta_data']['parameters']['optimize_input'] = self.optimize_input
            result['meta_data']['parameters']['gcd'] = self.gcd

            result['qubo'] = self.data_getter.get_qubo_result(result)
            return result
        else:
            return None

    def _qubo2maxcut(self, qubo: np.ndarray) -> np.ndarray:
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

        if self.optimize_input:
            unique_fields = set(q_sym.astype(int).flatten())
            greatest_common_divisor = reduce(gcd, unique_fields)
            if greatest_common_divisor > 1:
                q_sym /= greatest_common_divisor
                self.gcd = greatest_common_divisor

        Qe_plus_c = -np.array([(np.sum(q_sym, 1))])
        np.fill_diagonal(q_sym, 0)

        return np.block([
            [q_sym, Qe_plus_c.T],
            [Qe_plus_c, np.zeros((1, 1))]
        ])


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
        self.add_argument('-o', '--output', help='set custom output file path')
        # time limit format taken from SLURM docs https://slurm.schedmd.com/sbatch.html
        self.add_argument('-t', '--time', default='0', type=self.parse_time_limit,
                          help='set running time limit; acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"')

    def get_time_limit(self):
        ...

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
        self.add_argument('-O', '--optimize', action='store_true',
                          help='Divide QUBO values by their GCD')


class ParserDWaveHeuristic(ParserQubo):
    def __init__(self):
        super().__init__(prog='biqbin_heuristic.py',
                         description='Biqbin QUBO solver with DWave heuristic')
        self.add_argument('-d', '--debug', action='store_true',
                          help='enable debug logs')
        self.add_argument('-i', '--info', action='store_true',
                          help='enable info logs')
