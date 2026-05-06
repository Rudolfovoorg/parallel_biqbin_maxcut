__version__ = '2.0.5'

from deprecated import deprecated
import numpy.typing as npt
import numpy as np
import logging

from biqbin.utils import check_matrix_validity_wrap, divide_matrix_by_gcd, heur_root_data_collector
from biqbin.biqbin_module import (BabNode, Problem,
                                  run,
                                  update_mc_lower_bound_solution,
                                  set_heuristic, goemans_williamson_heuristic,
                                  set_node_evaluation, sdp_bound,
                                  get_rank)

# Initialize MPI at start
# https://stackoverflow.com/questions/7016056/python-logging-not-outputting-anything
logging.basicConfig()
logger = logging.getLogger(__name__)


class PrettyPrint:
    def __str__(self) -> str:
        return (f'class: {type(self).__name__}')

    def __repr__(self) -> str:
        return str(self)


class ProblemMaxCut(PrettyPrint):
    def __init__(self,
                 maxcut_adjacency_matrix: npt.NDArray[np.floating | np.integer],
                 problem_name: str,
                 optimize_mc_adj_matrix: bool = False) -> None:

        self.problem_name = problem_name
        self.optimize_mc_adj_matrix = optimize_mc_adj_matrix
        self.gcd: int = 1
        self.maxcut_adjacency_matrix = maxcut_adjacency_matrix

    @property
    @check_matrix_validity_wrap
    def maxcut_adjacency_matrix(self) -> npt.NDArray[np.floating | np.integer]:
        """Returns the valid input which Biqbin can solve.
        Checks if the input is valid for Biqbin (if all values are integers).

        Returns:
            np.ndarray: adjacency matrix for the MaxCut problem
        """

        return self._maxcut_adjacency_matrix

    @maxcut_adjacency_matrix.setter
    def maxcut_adjacency_matrix(self, value: npt.NDArray[np.floating | np.integer]):
        """Sets Biqbin input which is an adjacency matrix for the MaxCut problem.
        Optionally optimizes the input (divides the values of the matrix by their greatest common divisor).

        Args:
            value (npt.NDArray[np.floating  |  np.integer]): MaxCut adjacency matrix
        """
        if self.optimize_mc_adj_matrix:
            self.gcd = divide_matrix_by_gcd(value)
        self._maxcut_adjacency_matrix = value

    def __str__(self) -> str:
        return (f'{super().__str__()}\n'
                f'Problem name = {self.problem_name}\n'
                f'MaxCut adjacency matrix {self.maxcut_adjacency_matrix.shape} =\n'
                f'{self.maxcut_adjacency_matrix}\n')


class ProblemQubo(ProblemMaxCut):
    """Inherits from ProblemMaxCut, takes a qubo np.ndarray and constructs the biqbin input (maxcut adjacency matrix)
    """

    def __init__(self, Q: np.ndarray, offset: float, problem_name: str, is_minimization: bool, optimize_input: bool = False):
        """Initialize the ProblemQubo

        Args:
            Q (np.ndarray): Qubo in a dense matrix.
            problem_name (str): Name (filename) of the problem instance.
            is_minimization (bool): If the objective is to minimize (True) or maximize (False)
            optimize_input (bool, optional): Divides the biqbin input (maxcut adjacency matrix) by the greatest common divisor of the values.
                                             Defaults to False.
        """
        self.Q: np.ndarray = Q
        self.offset: float = offset
        self.is_minimization: bool = is_minimization
        if self.is_minimization:
            super().__init__(self.qubo2maxcut(Q), problem_name, optimize_input)
        else:
            super().__init__(self.qubo2maxcut(-Q), problem_name, optimize_input)

    @check_matrix_validity_wrap
    def qubo2maxcut(self, qubo: np.ndarray) -> np.ndarray:
        """Convert qubo to adjacency matrix that biqbin can read.
        Checks if the input qubo is valid (all values integers).

        Args:
            qubo (np.ndarray): qubo as 2d numpy array
        Returns:
            np.ndarray: adjacency matrix for max cut problem
        """

        q_sym = 1 / 2 * (qubo.T + qubo)

        Qe_plus_c = -np.array([(np.sum(q_sym, 1))])
        np.fill_diagonal(q_sym, 0)

        return np.block([
            [q_sym, Qe_plus_c.T],
            [Qe_plus_c, np.zeros((1, 1))]
        ])

    def __str__(self) -> str:
        return (f'{super().__str__()}\n'
                f'Problem name = {self.problem_name}\n'
                f'Offset = {self.offset}\n'
                f'Minimizing = {self.is_minimization}\n'
                f'Qubo {self.Q.shape} =\n{self.Q}\n')


class SolutionMaxCut(PrettyPrint):
    def __init__(self, biqbin_result: dict, problem: ProblemMaxCut) -> None:
        if problem.optimize_mc_adj_matrix and problem.gcd != 1:
            biqbin_result['maxcut']['computed_val'] *= problem.gcd

        self.__problem: ProblemMaxCut = problem
        self.__solution: dict = biqbin_result['maxcut']
        self.meta_data: dict = biqbin_result['meta_data']

    @property
    def solution(self) -> dict:
        return self.__solution

    @property
    def problem(self) -> ProblemMaxCut:
        return self.__problem

    def print_computed_solution(self, verbose: bool = False) -> None:
        print(f'{super().__str__()}\n'
              f'Problem name          = {self.meta_data['instance']}\n'
              f'Compute time          = {self.meta_data['time']:.2f} seconds\n'
              f'Time limit reached    = {self.meta_data['time_limit_reached']}\n'
              f'{self._get_verbose_metadata_string(verbose)}'
              f'--- Max-Cut ---\n'
              f' Computed value = {self.__solution['computed_val']}\n'
              f'Solution MaxCut = {self.__solution['solution']}\n'
              f'              x = {self.__solution['x']}\n')

    def _get_verbose_metadata_string(self, verbose: bool) -> str:
        if verbose:
            return (f'B&B nodes evaluated   = {self.meta_data['eval_bab_nodes']}\n'
                    f'Heurist run count     = {self.meta_data['heuristic_run_count']}\n'
                    f'Optimized input       = {self.meta_data['parameters']['optimized']}\n'
                    f'gcd = {self.meta_data['parameters']['gcd']}\n'
                    f'Worker processes used = {self.meta_data['num_workers_used']}\n')
        else:
            return ''

    def __str__(self) -> str:
        return (
            f'{super().__str__()}\n'
            f'   Problem name = {self.problem.problem_name}\n'
            f' Computed value = {self.__solution['computed_val']}\n'
            f'       Solution = {self.__solution['solution']}\n'
            f'              x = {self.__solution['x']}'
        )


class SolutionQubo(SolutionMaxCut):
    def __init__(self, biqbin_result: dict, problem: ProblemQubo) -> None:
        super().__init__(biqbin_result, problem)
        self.solution_maxcut = super().solution

        self.__problem: ProblemQubo = problem
        self.__solution = self.maxcut2qubo(problem)

    @property
    def solution(self):
        return self.__solution

    @property
    def problem(self) -> ProblemQubo:
        return self.__problem

    def maxcut2qubo(self, problem: ProblemQubo) -> dict:
        """Write qubo solution from the retrievied maxcut solution

        Returns:
            dict: Qubo specific result (computed_val, x, solution, cardinality, obj)
        """
        qubo_solution, qubo_x = self._maxcut_solution2qubo_solution(
            self.solution_maxcut["solution"]
        )

        computed_val = float(problem.Q.dot(
            qubo_x).dot(qubo_x)) + self.problem.offset
        return {'computed_val': computed_val,
                'offset': self.problem.offset,
                'solution': qubo_solution,
                'x': qubo_x,
                'cardinality': float(sum(qubo_x)),
                'minimization': problem.is_minimization
                }

    def _maxcut_solution2qubo_solution(self, maxcut_solution: np.ndarray):
        """Convert maxcut solution nodes to qubo solution nodes

        Args:
            maxcut_solution (np.ndarray): maxcut solution found by biqbin
        Returns:
            np.ndarray: qubo solution nodes
        """

        n, _ = self.problem.Q.shape

        _x_mc = np.array(maxcut_solution, dtype=int) - 1
        x_mc_sol = np.ones(n + 1)
        xx = np.zeros(n + 1, dtype=int)
        xx[_x_mc] = 1

        x_mc_sol[_x_mc] = -1
        x_mc_sol *= -x_mc_sol[-1]
        y = 1 / 2 * (x_mc_sol + 1)[:-1]
        qubo_solution = np.nonzero(y)[0] + 1

        return qubo_solution.tolist(), y.astype(int).tolist()

    def print_computed_solution(self, verbose: bool = False) -> None:
        if verbose:
            super().print_computed_solution()
        else:
            print(f'class: {type(self).__name__}\n'
                  f'   Problem name = {self.meta_data['instance']}\n'
                  f'   Compute time = {self.meta_data['time']:.2} seconds\n')
        print(
            f'--- QUBO ---\n'
            f' Computed value = {self.__solution['computed_val']}\n'
            f'       Solution = {self.__solution['solution']}\n'
            f'              x = {self.__solution['x']}\n'
            f'is_minimization = {self.__problem.is_minimization}'
        )

    def __str__(self):
        return (
            f'{super().__str__()}\n'
            f'   Problem name = {self.problem.problem_name}\n'
            f' Computed value = {self.__solution['computed_val']}\n'
            f'       Solution = {self.__solution['solution']}\n'
            f'              x = {self.__solution['x']}\n'
            f'is_minimization = {self.__problem.is_minimization}'
        )


class MaxCutSolver(PrettyPrint):
    """Default MaxCut Biqbin Python Wrapper, solves the MaxCut problem.
    """
    solver_name = f'PyBiqBin-MaxCut {__version__}'

    def __init__(self,
                 problem: ProblemMaxCut,
                 params: str = 'params',
                 time_limit: int = 0,
                 initial_estimate: np.ndarray | None = None,
                 collect_heuristic_data: bool = False):
        """Initialize the solver

        Args:
            params (str): path to parameters file
            time_limit (int): time limit in seconds
        """
        self.__problem: ProblemMaxCut = problem
        self.params: str = params
        self.time_limit: int = time_limit
        self.initial_estimate_solution = None

        self.rank: int = get_rank()
        if self.rank == 0:
            if initial_estimate is not None:
                self._check_solution_validity(
                    initial_estimate, problem.maxcut_adjacency_matrix.shape[0])
                self.initial_estimate_solution = initial_estimate
                self.root_heuristic = self._use_initial_estimate_on_root

            self._heuristic_fn = self.root_heuristic
            self._upper_bound_fn = self.root_upper_bound
        
        else:
            self._heuristic_fn = self.heuristic
            self._upper_bound_fn = self.upper_bound

        # Heuristic data collection
        self.collect_heuristic_root_data: bool = collect_heuristic_data
        self.heuristic_root_data = []
        
        set_node_evaluation(self._bab_node_evaluation)
        set_heuristic(self._call_heuristic)

        # SDPBound calls heuristic by itself, but if that is not called, we need to call it manually
        self._heuristic_was_called: bool = False

    @property
    def problem(self) -> ProblemMaxCut:
        return self.__problem

    def _bab_node_evaluation(self, node: BabNode, P0: Problem, P: Problem) -> float:
        """ Calls the upper and lower bound functions on a given BabNode

        Args:
            node (BabNode): current node
            P0 (Problem): Main-Problem
            P (Problem): Sub-Problem

        Returns:
            float: _description_
        """
        self._heuristic_was_called = False
        upper_bound_value = self._call_upper_bound(node, P0, P)
        if not self._heuristic_was_called:
            lower_bound_value = self._call_heuristic(
                P0.L, P.L, node.xfixed, node.sol.X)
        return upper_bound_value

    def upper_bound(self, node, P0, P, *args, **kwargs) -> float:
        logger.info('CALLED UPPER BOUND')
        return sdp_bound(node, P0, P)

    def root_upper_bound(self, node, P0, P, *args, **kwargs) -> float:
        logger.info('--- CALLED ROOT UPPER BOUND ---')
        return self.upper_bound(node, P0, P, *args, **kwargs)

    def heuristic(self, L: np.ndarray, **kwargs) -> npt.ArrayLike:
        """
        Args:
            L (np.ndarray): Sub-Problem Laplacean matrix

        Returns:
            float: value of the solution array "x" found by the heuristic function
        """
        logger.info("CALLED HEURISTIC")
        
        L0 = kwargs['L0']
        xfixed = kwargs['xfixed']
        sol_X = kwargs['sol_X']
        # biqbin gw expects a full solution vector
        x = np.zeros(L0.shape[0] - 1, dtype=np.int32)

        goemans_williamson_heuristic(
            L0, L, xfixed, sol_X, x
        )

        return x[xfixed == 0]

    def root_heuristic(self, L: np.ndarray, **kwargs) -> npt.ArrayLike:
        logger.info("--- CALLED ROOT HEURISTIC ---")
        return self.heuristic(L, **kwargs)

    def _call_upper_bound(self, node, P0, P, *args, **kwargs) -> float:
        """Wrapper to check if check if check
        """
        return self._upper_bound_fn(node, P0, P, *args, **kwargs)

    @heur_root_data_collector()
    def _call_heuristic(self, L0: np.ndarray, L: np.ndarray, xfixed: np.ndarray, sol_X: np.ndarray) -> float:
        self._heuristic_was_called = True

        # Call heuristic function get the solution vector
        heur_sol = self._heuristic_fn(L, L0=L0, xfixed=xfixed, sol_X=sol_X)

        # Check if the solution is in valid format
        if not isinstance(heur_sol, np.ndarray):
            heur_sol = np.array(heur_sol)

        self._check_solution_validity(heur_sol, L.shape[0] - 1)

        # copy to full solution x
        x = sol_X.copy()
        x[xfixed == 0] = heur_sol

        heur_value = self._evaluate_solution(L0, x)
        solution_updated = update_mc_lower_bound_solution(x)

        if logger.isEnabledFor(logging.DEBUG):
            default_gw_value = goemans_williamson_heuristic(
                L0, L, xfixed, sol_X, np.zeros(L0.shape[0])
            )
            logger.debug(
                f'Custom heuristic: {heur_value}; default gw heuristic: {default_gw_value}'
            )

        return heur_value

    def _use_initial_estimate_on_root(self, L: np.ndarray, **kwargs) -> npt.ArrayLike:
        """ heuristic call on root node if initial estimate solution is passed in
        """
        if kwargs['L0'].shape != L.shape:
            raise ValueError(
                f"Main problem shape {kwargs['L0'].shape} != Subproblem shape {L.shape}!")
        if kwargs['L0'].shape != self.problem.maxcut_adjacency_matrix.shape:
            raise ValueError(
                f"Main problem shape {kwargs['L0'].shape} != mc adjacency matrix shape {self.problem.maxcut_adjacency_matrix.shape}!")
        if np.any(kwargs['xfixed']):
            raise ValueError("xfixed is nonzero!")

        # Biqbin solution has 1 element less than the problem size, the last elemenent is assumed to be 0
        MC_DUMMY_ELEMENT_COUNT = 1

        return self.initial_estimate_solution[:-MC_DUMMY_ELEMENT_COUNT] # pyright: ignore[reportOptionalSubscript]


    def _evaluate_solution(self, L0: np.ndarray, sol: np.ndarray) -> float:
        """Calculate the Max-Cut lower bound value of the heuristic solution

        Args:
            L0 (np.ndarray): main Problem *SP->L matrix
            sol (np.ndarray): current solution

        Returns:
            float: value of the solution
        """
        self._check_solution_validity(
            sol, self.problem.maxcut_adjacency_matrix.shape[0] - 1)
        sol_val = sol @ L0[:-1, :-1] @ sol

        return float(sol_val)

    def _run_solver(self) -> dict | None:
        """Runs Biqbin C/C++ implementation

        Raises:
            ValueError: If no result is retrieved on the master process.

        Returns:
            dict | None: Solution python dict built by C++, or None on MPI rank != 0
        """
        if self.problem is None:
            raise ValueError("Problem instance not set!")

        if self.rank == 0:
            input_matrix = self.problem.maxcut_adjacency_matrix.astype(
                np.float64)
            logger.info(f'Solving {self.problem}')
        else:
            input_matrix = None

        biqbin_result = run(self.solver_name,
                            self.problem.problem_name,
                            input_matrix,
                            self.params,
                            self.time_limit)

        if self.rank == 0:
            if biqbin_result is None:
                raise ValueError(
                    'Result from BiqBin is None, computation failed!')

            biqbin_result['meta_data']['solver'] = self.solver_name
            biqbin_result['meta_data']['instance'] = self.problem.problem_name
            biqbin_result['meta_data']['parameters'] = {
                'time_limit': self.time_limit if self.time_limit > 0 else None,
                'optimized': self.problem.optimize_mc_adj_matrix,
                'gcd': self.problem.gcd
            }
            if self.collect_heuristic_root_data:
                biqbin_result['meta_data']['root_node']['total_heur_time'] = sum(
                    d['time'] for d in self.heuristic_root_data)
                biqbin_result['meta_data']['root_node']['heuristic_data'] = self.heuristic_root_data
            return biqbin_result

    def compute(self) -> SolutionMaxCut | None:
        """Compute the MaxCut solution using Biqbin

        Args:
            problem (ProblemMaxCut): problem to be solved.

        Returns:
            SolutionMaxCut | None: Returns the solution class on MPI rank == 0.
        """
        biqbin_result = self._run_solver()
        if biqbin_result is not None:
            return SolutionMaxCut(biqbin_result, self.problem)
        else:
            return None

    def _check_solution_validity(self, initial_solution: np.ndarray, problem_size: int):
        if initial_solution.ndim != 1 or initial_solution.shape[0] != problem_size:
            raise ValueError(
                f"Solution must be a 1D vector of size {problem_size}, but got shape {initial_solution.shape}!")

        if not ((initial_solution == 0) | (initial_solution == 1)).all():
            raise ValueError("Solution must be a binary vector!")

    def __str__(self) -> str:
        return (f'{super().__str__()}'
                f'    Problem = {self.problem.problem_name}\n'
                f'Params path = {self.params}\n'
                f' Time limit = {self.time_limit}\n')


class QUBOSolver(MaxCutSolver):
    solver_name = f'PyBiqBin-QUBO {__version__}'

    def __init__(self,
                 problem: ProblemQubo,
                 params: str = 'params',
                 time_limit: int = 0,
                 initial_estimate: np.ndarray | None = None,
                 collect_heur_data: bool = False):

        mc_initial_solution = None
        if get_rank() == 0 and initial_estimate is not None:
            self._check_solution_validity(
                initial_estimate, problem.Q.shape[0])
            mc_initial_solution = np.append(initial_estimate, 0)

        super().__init__(problem, params, time_limit,
                         mc_initial_solution, collect_heur_data)

        self.__problem: ProblemQubo = problem

    @property
    def problem(self) -> ProblemQubo:
        return self.__problem

    def compute(self) -> SolutionQubo | None:
        """Computes the solution to the QUBO using Biqbin, only MPI rank == 0 returns Solution

        Returns:
            SolutionQubo: Solution class for QUBO problem. returns None if MPI rank != 0.
        """
        biqbin_result = self._run_solver()

        if biqbin_result is not None:
            return SolutionQubo(biqbin_result, self.problem)
        else:
            return None
