__version__ = '2.0.0'

from abc import ABC, abstractmethod
import argparse
from functools import reduce
from math import gcd
import numpy as np
import scipy as sp
import json
from glob import glob

from biqbin import (run, set_heuristic,
                    default_heuristic,
                    get_rank, set_read_data,
                    default_read_data)


class DataGetter(ABC):
    """Abstract class to parse qubo data
    """
    @abstractmethod
    def problem_instance_name(self) -> str:
        ...

    @abstractmethod
    def problem_instance(self):
        ...
        
    @abstractmethod
    def read_file(self):
        ...
        
    def from_sparse(self, sparse_matrix):
        """Converts from sparse coo matrix to regular form

        Args:
            sparse_matrix (dict): scipy sparse coo matrix

        Returns:
            np.ndarray: regular form matrix
        """
        return sp.sparse.coo_matrix(
            (sparse_matrix['data'], (sparse_matrix['row'], sparse_matrix['col'])),
            shape=sparse_matrix['shape'], dtype='float'
        ).todense().getA()


class DataGetterMaxCutDefault(DataGetter):
    """
    Uses the default C implementation or MaxCut, reads and parses maxcut instance file in edge weight list format and parses
    into the adjacency matrix.
    """
    def __init__(self, filename: str):
        self.filename = filename
        self.adj_matrix = None
    
    def problem_instance_name(self) -> str:
        """Get the instance file path

        Returns:
            str: path to instance file
        """
        return self.filename

    def problem_instance(self):
        """Gets the adjacency matrix from the instance file
        """
        return self.adj_matrix
    
    def read_file(self):
        self.adj_matrix = default_read_data(self.filename)
        return self.adj_matrix
    

class DataGetterAdjacencyJson(DataGetterMaxCutDefault):
    """
    DataGetter for the Maxcut class, reads and parses json serialized dict with adj key and sparse coo matrix as value.
    """
    def read_file(self):
        with open(self.filename, "r") as f:
            mc_data = json.load(f)

        self.adj_matrix = self.from_sparse(mc_data["adjacency"])
        
        adj_int = np.array(self.adj_matrix, dtype=np.int64)                
        if not np.all(self.adj_matrix == adj_int):
            raise ValueError("All values in the adjacency matrix need to be integers!")
        
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
        # For testing purposes
        self.heuristic_counter = 0
        self.time_limit = time_limit

    def read_data(self) -> np.ndarray:
        """Transform edge weight list into an adjacancy matrix

        Returns:
            np.ndarray: adjacency matrix
        """
        return self.data_getter.read_file()

    def heuristic(self, L0: np.ndarray, L: np.ndarray, xfixed: np.array, sol_X: np.array, x: np.array) -> float:
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
        self.heuristic_counter += 1
        return default_heuristic(L0, L, xfixed, sol_X, x)

    def run(self):
        """Runs Biqbin Maxcut solver
        
        Returns:
            dict: result dict with keys: "max_val" - max cut solution value, "solution" - nodes in this solution, "time" - spent solving 
        """
        result = run(self.solver_name, self.data_getter.problem_instance_name(), self.params, self.time_limit)
        if (self.get_rank() == 0):
            result['maxcut']['solution'] = result['maxcut']['solution'].tolist()
            result['meta_data']['instance'] = self.data_getter.problem_instance_name()
            result['meta_data']['parameters'] = { 'time_limit': self.time_limit if self.time_limit > 0 else None }
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


class DataGetterJson(DataGetter):
    """Reads qubo instance file, should be a json dictionary with "qubo" key
    and a COO sparse matrix with data, row and col. Indices starts from zero.
    """

    def __init__(self, filename: str):
        """Load data from json file and save the data and qubo

        Args:
            filename (str): path to file
        """
        self.filename = filename

    def problem_instance_name(self) -> str:
        """Get the instance file path

        Returns:
            str: path to instance file
        """
        return self.filename

    def problem_instance(self) -> np.ndarray:
        """Gets the qubo

        Returns:
            nd.ndarray: qubo in a numpy array
        """
        return self.qubo
    
    def read_file(self):
        with open(self.filename, "r") as f:
            self.qubo_data = json.load(f)
        self.qubo = self.from_sparse(self.qubo_data["qubo"])
        return self.qubo


class QUBOSolver(MaxCutSolver):
    solver_name = f'PyBiqBin-QUBO {__version__}'

    def __init__(self, data_getter: DataGetter, params: str, optimize_input:bool=False, time_limit:int=0):
        super().__init__(data_getter, params, time_limit)
        self.optimize_input: bool = optimize_input
        self.gcd: int = 1

    def _qubo2maxcut(self, qubo: np.ndarray) -> np.ndarray:
        """Convert qubo to adjacency matrix that biqbin can read

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

    def _maxcut_solution2qubo_solution(self, maxcut_solution: np.ndarray):
        """Convert maxcut solution nodes to qubo solution node

        Args:
            maxcut_solution (np.ndarray): maxcut solution found by biqbin
        Returns:
            np.ndarray: qubo solution nodes
        """

        n, _ = self.data_getter.problem_instance().shape

        _x_mc = np.array(maxcut_solution, dtype=int)-1
        x_mc_sol = np.ones(n + 1)
        xx = np.zeros(n + 1, dtype=int)
        xx[_x_mc] = 1

        x_mc_sol[_x_mc] = -1
        x_mc_sol *= -x_mc_sol[-1]
        x_mc_sol
        y = 1/2*(x_mc_sol+1)[:-1]
        qubo_solution = np.nonzero(y)[0] + 1
        return qubo_solution.tolist(), y.astype(int).tolist(), xx.tolist()

    def read_data(self) -> np.ndarray:
        """Read qubo json file, return an adjacency matrix for maxcut

        Returns:
            np.ndarray: adjacency matrix
        """
        return self._qubo2maxcut(self.data_getter.read_file())

    def run(self) -> dict:
        """Runs the original biqbin then adds the qubo solution nodes to the result dict

        Returns:
            dict: result dict containing "maxcut" and "qubo" keys with their respective solutions
        """
        result = super().run()
        if (self.get_rank() == 0):
            qubo_solution, qubo_x, mc_x = self._maxcut_solution2qubo_solution(result["maxcut"]["solution"])
            computed_val = self.data_getter.problem_instance().dot(qubo_x).dot(qubo_x)
            cardinality = sum(qubo_x)
            result['maxcut']['computed_val'] *= self.gcd
            result['maxcut']['x'] = mc_x
            result['qubo'] = {'computed_val': float(computed_val),
                             'solution': qubo_solution,
                             'x': qubo_x,
                             'cardinality': float(cardinality),
                             }
            result['meta_data']['parameters']['optimize_input'] = self.optimize_input
            result['meta_data']['parameters']['gcd'] = self.gcd
            return result
        else:
            return None
        

class BaseParser(argparse.ArgumentParser):
    def __init__(self, prog: str, description: str):
        super().__init__(prog=prog, description=description,
        usage=f'mpirun [-n N] python3 {prog} problem_instance [-p PARAMS] [-w] [-o OUTPUT]',
        epilog='For more information please visit https://github.com/Rudolfovoorg/parallel_biqbin_maxcut',
    )
        self.add_argument('problem_instance', help='Path to the problem instance file')
        
        # Optional arguments
        self.add_argument('-p', '--params', default='params', help='custom parameters file path (default: "params")')
        self.add_argument('-w', '--overwrite', 
                        action='store_true',
                        help='overwrite output.json instead of labeling with _NUMBER'
                        )
        self.add_argument('-o', '--output', help='set custom output file path')
        # time limit format taken from SLURM docs https://slurm.schedmd.com/sbatch.html
        self.add_argument('-t', '--time', default='0', type=self.parse_time_limit, help='set running time limit; acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"')
    
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
        self.add_argument('-e', '--edge_weight', action='store_true', help='use edge weight input file')

        
class ParserQubo(BaseParser):
    def __init__(self, prog=f'biqbin_qubo.py', description='Biqbin QUBO solver'):
        super().__init__(prog=prog, description=description)
        self.add_argument('-O', '--optimize', action='store_true', help='Divide QUBO values by their GCD')
        
class ParserDWaveHeuristic(ParserQubo):
    def __init__(self):
        super().__init__(prog='biqbin_heuristic.py', description='Biqbin QUBO solver with DWave heuristic')
        self.add_argument('-d', '--debug', action='store_true', help='enable debug logs')
        self.add_argument('-i', '--info', action='store_true', help='enable info logs')