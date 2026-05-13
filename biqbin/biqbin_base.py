__version__ = '2.0.5'

from typing import ClassVar

from deprecated import deprecated
import numpy.typing as npt
import numpy as np
import logging

import biqbin
from biqbin import biqbin_module
from biqbin.utils import check_matrix_validity_wrap, divide_matrix_by_gcd, data_collector
from biqbin.biqbin_module import (BabNode, Problem, abort_mpi,
                                  run,
                                  update_mc_lower_bound_solution,
                                  set_heuristic, goemans_williamson_heuristic,
                                  set_primal_solution, set_node_evaluation, sdp_bound,
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
    solver_name: ClassVar[str] = f'PyBiqBin-MaxCut {__version__}'

    # Biqbin solution has 1 element less than the problem size, the last elemenent is assumed to be 0
    _MC_DUMMY_ELEMENT: ClassVar[int] = 1

    def __init__(self,
                 problem: ProblemMaxCut,
                 params: str = 'params',
                 time_limit: int = 0,
                 initial_estimate: np.ndarray | None = None,
                 collect_heuristic_data: bool = False,
                 collect_sdp_bound_root_data: bool = True):
        """Initialize the solver

        Args:
            problem (ProblemMaxCut): Problem instance to be solved
            params (str): path to parameters file
            time_limit (int): time limit in seconds
            initial_estimate (np.ndarray or None): Optional initial estimate solution binary vector
            collect_heuristic_data (bool): If True saves runtimes of each heuristic call and the objective value of the found solution
        """
        self.__problem: ProblemMaxCut = problem
        self.params: str = params
        self.time_limit: int = time_limit
        self.initial_estimate_solution: np.ndarray | None = None

        self.rank: int = get_rank()
        # On root rank
        if self.rank == 0:
            if initial_estimate is not None:
                # Use the initial estimate solution on the root node heuristic call
                self._check_solution_validity(
                    initial_estimate, problem.maxcut_adjacency_matrix.shape[0])
                self.initial_estimate_solution = initial_estimate
                self.root_heuristic = self._use_initial_estimate_on_root
                logger.info(
                    f'Using an initial estimate solution: {self.initial_estimate_solution}')

            self._heuristic_fn = self.root_heuristic
            self._sdp_bound_fn = self.root_sdp_bound

        # On worker rank
        else:
            # Use the non-root heuristic and upper_bound methods on worker processes
            self._heuristic_fn = self.heuristic
            self._sdp_bound_fn = self.sdp_bound

        # Heuristic data collection
        self.collect_heuristic_root_data: bool = collect_heuristic_data if self.rank == 0 else False
        self.heuristic_root_data: list[dict] = []
        self.collect_sdp_bound_root_data: bool = collect_sdp_bound_root_data if self.rank == 0 else False
        self.sdp_bound_root_data: list[dict] = []

        # Set the functions in C++ source to be called
        set_node_evaluation(self._bab_node_evaluation)
        set_heuristic(self._call_heuristic)

        # biqbin_module.sdp_bound needs to be called before goemans_williams_heuristic is called
        self._primal_solution_set: bool = False
        # SDPBound calls heuristic by itself, if that is overwritten, we need to call it manually
        self._heuristic_was_called: bool = False

    @property
    def problem(self) -> ProblemMaxCut:
        return self.__problem

    def compute(self) -> SolutionMaxCut | None:
        """Compute the MaxCut solution using Biqbin

        Returns:
            SolutionMaxCut | None: Returns the solution class on MPI rank == 0.
        """
        biqbin_result = self._run_solver()
        if biqbin_result is not None:
            return SolutionMaxCut(biqbin_result, self.problem)
        else:
            return None

    def sdp_bound(self, node: BabNode, P0: Problem, P: Problem) -> float:
        """Compute the SDP bound for a B&B node.

        Override in subclasses to use a custom bounding strategy.
        """
        logger.debug(f'[rank {self.rank}]: CALLED UPPER BOUND')
        self._primal_solution_set = True
        return sdp_bound(node, P0, P)

    def root_sdp_bound(self, node: BabNode, P0: Problem, P: Problem, *args, **kwargs) -> float:
        """Compute the SDP bound on the root node. 
        By default it will stay the same as self.upper_bound computed on
        non-root nodes. 

        Override in subclasses to use a custom bounding strategy.
        """
        return self.sdp_bound(node, P0, P, *args, **kwargs)

    def heuristic(self, L: np.ndarray, *args, **kwargs) -> npt.ArrayLike:
        """
        Finds a heuristic solution binary vector for the subproblem Laplacean matrix L.

        Args:
            L (np.ndarray): Subproblem Laplacean matrix for the Max-Cut Problem

        Keyword Args:
            L0 (np.ndarray): Laplacian of the full (original) problem.
            xfixed (np.ndarray): Fixed-variable mask, shape (n,).
            sol_X (np.ndarray): Stores the value of the fixed variables, shape (n,).

        Returns:
            np.ndarray: Binary solution vector found by the heuristic. Shape must be a 1D of length one less than Subproblem size (L.shape[0] - 1, )
        """
        P0 = kwargs['P0']
        P = kwargs['P']
        node = kwargs['node']

        if not self._primal_solution_set:
            logger.fatal(
                f'SDP primal solution was not set {self.rank}!')
            abort_mpi(10)

        # biqbin GW implementation expects a full solution vector
        x = np.zeros(P0.n - MaxCutSolver._MC_DUMMY_ELEMENT, dtype=np.int32)

        # solution is stored in x where solution variables are not fixed
        goemans_williamson_heuristic(P0.L, L, node.xfixed, node.sol.X, x)

        # return only where solution variables are not fixed
        return x[node.xfixed == 0]

    def root_heuristic(self, L: np.ndarray, **kwargs) -> npt.ArrayLike:
        return self.heuristic(L, **kwargs)

    def set_sdp_primal_solution(self, primal_solution: np.ndarray):
        set_primal_solution(primal_solution)
        self._primal_solution_set = True
    """
    ####################################################
                    Private
    ####################################################
    """

    def _bab_node_evaluation(self, node: BabNode, P0: Problem, P: Problem, *args, **kwargs) -> float:
        """ Compute the upper and lower bound of the current B&B node

        Args:
            node (BabNode): current B&B node
            P0 (Problem): Original (full) problem
            P (Problem): current nodes subproblem

        Returns:
            float: sdp bound value of the current node
        """
        # Check if the primal solution was set for the default GW heuristic,
        # default sdp_bound does this by itself.
        self._primal_solution_set = False

        # Biqbin's default sdp_bound calls heuristic function
        # but if we use a different sdp functions we need to call it manually
        self._heuristic_was_called = False

        upper_bound_value = self._call_sdp_bound(node, P0, P)
        if not self._heuristic_was_called:
            self._call_heuristic(node, P0, P)
        return upper_bound_value

    @data_collector(enabled_flag='collect_sdp_bound_root_data', data_box='sdp_bound_root_data')
    def _call_sdp_bound(self, node: BabNode, P0: Problem, P: Problem) -> float:
        """Wrapper around sdp bound function call in case we want to add other data
        """
        return self._sdp_bound_fn(node, P0, P)

    @data_collector(enabled_flag='collect_heuristic_root_data', data_box='heuristic_root_data')
    def _call_heuristic(self, node: BabNode, P0: Problem, P: Problem) -> float:
        # Call heuristic function
        heur_sol = self._heuristic_fn(P.L, node=node, P0=P0, P=P)
        self._heuristic_was_called = True

        # Check if the solution is in valid format
        if not isinstance(heur_sol, np.ndarray):
            heur_sol = np.array(heur_sol)
        self._check_solution_validity(
            heur_sol, P.n - MaxCutSolver._MC_DUMMY_ELEMENT)

        # copy to full solution x
        x = node.sol.X.copy()
        x[node.xfixed == 0] = heur_sol

        heur_value = self._evaluate_solution(P0.L, x)
        solution_updated = update_mc_lower_bound_solution(x)

        if logger.isEnabledFor(logging.DEBUG):
            default_gw_value = goemans_williamson_heuristic(
                P0.L, P.L, node.xfixed, node.sol.X, np.zeros(P0.n)
            )
            logger.debug(
                f'Custom heuristic: {heur_value}; default gw heuristic: {default_gw_value}'
            )

        return heur_value

    def _use_initial_estimate_on_root(self, L: np.ndarray, *args, **kwargs) -> npt.ArrayLike:
        """ heuristic call on root node if initial estimate solution is passed in
        """
        if logger.isEnabledFor(logging.DEBUG):
            if kwargs['L0'].shape != L.shape:
                logger.fatal(
                    f"Main problem shape {kwargs['L0'].shape} != Subproblem shape {L.shape}!")
                abort_mpi(10)
            if kwargs['L0'].shape != self.problem.maxcut_adjacency_matrix.shape:
                logger.fatal(
                    f"Main problem shape {kwargs['L0'].shape} != mc adjacency matrix shape {self.problem.maxcut_adjacency_matrix.shape}!")
                abort_mpi(10)
            if np.any(kwargs['xfixed']):
                logger.fatal("xfixed is nonzero!")
                abort_mpi(10)

        return self.initial_estimate_solution[:-MaxCutSolver._MC_DUMMY_ELEMENT] # pyright: ignore[reportOptionalSubscript]

    def _evaluate_solution(self, L0: np.ndarray, sol: np.ndarray) -> float:
        """Calculate the Max-Cut objective value of the solution

        Args:
            L0 (np.ndarray): Laplacean of the original (full) problem
            sol (np.ndarray): current solution

        Returns:
            float: value of the solution
        """
        self._check_solution_validity(
            sol, self.problem.maxcut_adjacency_matrix.shape[0] - MaxCutSolver._MC_DUMMY_ELEMENT)
        sol_val = sol @ L0[:-MaxCutSolver._MC_DUMMY_ELEMENT,
                           :-MaxCutSolver._MC_DUMMY_ELEMENT] @ sol

        return float(sol_val)

    def _run_solver(self) -> dict | None:
        """Runs Biqbin C/C++ implementation

        Returns:
            dict: Solution python dict built by C++, or empty dict on MPI rank != 0
        """
        # MC adjacency matrix is passed in on rank 0 and broadcasted to worker ranks during execution
        if self.rank == 0:
            input_matrix = self.problem.maxcut_adjacency_matrix.astype(
                np.float64)
            logger.info(f'Solving {self.problem}')
        else:
            input_matrix = None

        raw_results = run(self.solver_name,
                          self.problem.problem_name,
                          input_matrix,
                          self.params,
                          self.time_limit)

        if self.rank == 0:
            return self._update_result_dict(raw_results)
        else:
            return None

    def _update_result_dict(self, result: dict):
        assert self.rank == 0, "_build_result must only be called on rank 0"

        result['meta_data']['solver'] = self.solver_name
        result['meta_data']['instance'] = self.problem.problem_name
        result['meta_data']['parameters'] = {
            'time_limit': self.time_limit if self.time_limit > 0 else None,
            'optimized': self.problem.optimize_mc_adj_matrix,
            'gcd': self.problem.gcd
        }
        if self.collect_heuristic_root_data:
            result['meta_data']['root_node']['total_heur_time'] = sum(
                d['time'] for d in self.heuristic_root_data)
            result['meta_data']['root_node']['heuristic_data'] = self.heuristic_root_data
        if self.collect_sdp_bound_root_data:
            result['meta_data']['root_node']['sdp_bound_data'] = self.sdp_bound_root_data
        return result

    def _check_solution_validity(self, initial_solution: np.ndarray, problem_size: int):
        fatal_error = False
        if initial_solution.ndim != 1 or initial_solution.shape[0] != problem_size:
            logger.fatal(
                f"Solution must be a 1D vector of size {problem_size}, but got shape {initial_solution.shape}!")
            fatal_error = True
        if not ((initial_solution == 0) | (initial_solution == 1)).all():
            logger.fatal("Solution must be a binary vector!")
            fatal_error = True

        if fatal_error:
            abort_mpi(10)

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
            # Maxcut solution is one larger than Qubo solution
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
