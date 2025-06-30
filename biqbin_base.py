__version__ = '1.0.0'

from abc import ABC, abstractmethod
import numpy as np
import scipy as sp
import json

from biqbin import (run, set_heuristic,
                    default_heuristic,
                    get_rank, set_read_data,
                    default_read_data)


class MaxCutSolver:
    """Default MaxCut Biqbin Wrapper, runs Biqbin MaxCut using its original functions
    """
    solver_name = f'PyBiqBin-MaxCut {__version__}'

    def __init__(self, problem_instance_name: str, params: str):
        """Initialize the solver

        Args:
            problem_instance_name (str): path to problem instance in edge weight list format
            params (str): path to parameters file
        """
        self.problem_instance_name = problem_instance_name
        self.params = params
        set_read_data(self.read_data)
        set_heuristic(self.heuristic)
        # For testing purposes
        self.heuristic_counter = 0

    def read_data(self) -> np.ndarray:
        """Transform edge weight list into an adjacancy matrix

        Returns:
            np.ndarray: adjacency matrix
        """
        result = default_read_data(self.problem_instance_name)
        return result

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
        result = run(self.solver_name, self.problem_instance_name, self.params)

        if (self.get_rank() == 0):
            result['maxcut']['solution'] = result['maxcut']['solution'].tolist()
            return result
        else:
            return None

    def get_rank(self) -> int:
        """MPI process rank

        Returns:
            int: rank
        """
        return get_rank()

    def save_result(self, result, output_path=None):
        """_summary_

        Args:
            result (dict): result dictionary returned by biqbin
            output_path (str, optional): custom path to an output file. Defaults to None.
        """
        output_path = self._process_output_path(output_path)
        with open(output_path, "w") as f:
            json.dump(result, f, default=self._convert_numpy)

    def _process_output_path(self, output_path: str) -> str:
        if output_path is None or output_path == self.problem_instance_name:
            return self.problem_instance_name + ".output.json"
        if not output_path.endswith(".json"):
            output_path += ".json"
        return output_path

    def _convert_numpy(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class DataGetter(ABC):
    """Abstract class to parse qubo data
    """
    @abstractmethod
    def problem_instance_name(self) -> str:
        ...

    @abstractmethod
    def problem_instance(self):
        ...


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
        with open(filename, "r") as f:
            self.qubo_data = json.load(f)
            self.qubo = self.from_sparse(self.qubo_data["qubo"])

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

    def from_sparse(self, qubo_sparse):
        """Converts qubo from sparse to regular form

        Args:
            qubo_sparse (dict): scipy sparse coo matrix represantation of the qubo

        Returns:
            np.ndarray: qubo in regular form
        """
        return sp.sparse.coo_matrix(
            (qubo_sparse['data'], (qubo_sparse['row'], qubo_sparse['col'])),
            shape=qubo_sparse['shape'], dtype='float'
        ).todense().getA()


class QUBOSolver(MaxCutSolver):
    solver_name = f'PyBiqBin-QUBO {__version__}'

    def __init__(self, data_getter: DataGetter, params: str):
        self.data_getter = data_getter
        super().__init__(data_getter.problem_instance_name(), params)

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
        return self._qubo2maxcut(self.data_getter.problem_instance())

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
            result['maxcut']['x'] = mc_x
            result['qubo'] = {'computed_val': float(computed_val),
                             'solution': qubo_solution,
                             'x': qubo_x,
                             'cardinality': float(cardinality)}
            return result
        else:
            return None
