__version__ = '2.0.5'

import numpy.typing as npt
import numpy as np
import logging

from biqbin.utils import PrettyPrint, heur_root_data_collector
from biqbin.biqbin_module import (run, set_heuristic, init_mpi,
                                  goemans_williamson_heuristic, get_rank)

from biqbin.parameters import BiqbinParameters
from biqbin.problem import ProblemMaxCut, ProblemQubo
from biqbin.solution import SolutionMaxCut, SolutionQubo

# Initialize MPI at start
init_mpi()
# https://stackoverflow.com/questions/7016056/python-logging-not-outputting-anything
logging.basicConfig()
logger = logging.getLogger(__name__)


class MaxCutSolver(PrettyPrint):
    """Default MaxCut Biqbin Python Wrapper, solves the MaxCut problem.
    """
    solver_name = f'PyBiqBin-MaxCut {__version__}'

    def __init__(self,
                 problem: ProblemMaxCut,
                 params: str | BiqbinParameters,
                 initial_estimate: np.ndarray | None = None,
                 collect_heuristic_data: bool = False):
        """Initialize the solver

        Args:
            params (str): BiqbinParameters instance or path to parameters.toml file
        """
        self.__problem: ProblemMaxCut = problem
        if isinstance(params, str):
            self.params: BiqbinParameters = BiqbinParameters.from_toml(params)
        else:
            self.params = params
        self.initial_estimate_solution = None

        self.rank: int = get_rank()
        if self.rank == 0 and initial_estimate is not None:
            self._check_solution_validity(
                initial_estimate, problem.maxcut_adjacency_matrix.shape[0])
            self.initial_estimate_solution = initial_estimate
            self.heuristic = self._use_initial_estimate_on_root

        # Heuristic data collection
        self.collect_heuristic_root_data: bool = collect_heuristic_data
        self.heuristic_root_data = []

        set_heuristic(self._call_heuristic)

    @property
    def problem(self) -> ProblemMaxCut:
        return self.__problem

    @heur_root_data_collector()
    def _call_heuristic(self, L0: np.ndarray, L: np.ndarray, xfixed: np.ndarray, sol_X: np.ndarray, x: np.ndarray) -> float:

        # Call heuristic function get the solution vector
        heur_sol = self.heuristic(L, L0=L0, xfixed=xfixed, sol_X=sol_X)

        # Check if the solution is in valid format
        if not isinstance(heur_sol, np.ndarray):
            heur_sol = np.array(heur_sol)

        self._check_solution_validity(heur_sol, L.shape[0] - 1)

        # copy to full solution x
        np.copyto(x, sol_X)
        x[xfixed == 0] = heur_sol

        heur_value = self._evaluate_solution(L0, x)

        if logger.isEnabledFor(logging.DEBUG):
            default_gw_value = goemans_williamson_heuristic(
                L0, L, xfixed, sol_X, np.zeros(L0.shape[0])
            )
            logger.debug(
                f'Custom heuristic: {heur_value}; default gw heuristic: {default_gw_value}'
            )

        return heur_value

    def heuristic(self, L: np.ndarray, **kwargs) -> npt.ArrayLike:
        """Default GW heuristic (heuristic_unpacked in heuristic.c)

        Args:
            L (np.ndarray): Subproblem Problem Laplacean matrix

        Returns:
            float: value of the solution array "x" found by the heuristic function
        """
        L0 = kwargs['L0']
        xfixed = kwargs['xfixed']
        sol_X = kwargs['sol_X']
        # biqbin gw expects a full solution vector
        x = np.zeros(L0.shape[0] - 1, dtype=np.int32)

        goemans_williamson_heuristic(
            L0, L, xfixed, sol_X, x
        )

        return x[xfixed == 0]

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
        
        return self.initial_estimate_solution[:-1] # pyright: ignore[reportOptionalSubscript]

    def _evaluate_solution(self, L0: np.ndarray, sol: np.ndarray) -> float:
        """Calculate the Max-Cut lower bound value of the heuristic solution

        Args:
            L0 (np.ndarray): main Problem *SP->L matrix
            sol (np.ndarray): current solution

        Returns:
            float: value of the solution
        """
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
            logger.debug(self.params)
            input_matrix = self.problem.maxcut_adjacency_matrix.astype(
                np.float64)
            print(f'Solving {self.problem}')
        else:
            input_matrix = None

        biqbin_result = run(self.solver_name,
                            self.problem.problem_name,
                            input_matrix,
                            self.params)

        if self.rank == 0:
            if biqbin_result is None:
                raise ValueError(
                    'Result from BiqBin is None, computation failed!')

            biqbin_result['meta_data']['solver'] = self.solver_name
            biqbin_result['meta_data']['instance'] = self.problem.problem_name
            biqbin_result['meta_data']['parameters'] = {
                'time_limit': self.params.time_limit if self.params.time_limit > 0 else None,
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

    def _check_solution_validity(self, initial_solution: np.ndarray, problem_size):
        if initial_solution.ndim != 1 or initial_solution.shape[0] != problem_size:
            raise ValueError(
                f"Solution must be a 1D vector of size {problem_size}, but got shape {initial_solution.shape}!")

        if not ((initial_solution == 0) | (initial_solution == 1)).all():
            raise ValueError("Solution must be a binary vector!")

    def __str__(self) -> str:
        return (f'{super().__str__()}'
                f'    Problem = {self.problem.problem_name}'
                f'Params path = {self.params}')


class QUBOSolver(MaxCutSolver):
    solver_name = f'PyBiqBin-QUBO {__version__}'

    def __init__(self,
                 problem: ProblemQubo,
                 params: str | BiqbinParameters = 'biqbin.toml',
                 initial_estimate: np.ndarray | None = None,
                 collect_heur_data: bool = False):

        mc_initial_solution = None
        if get_rank() == 0 and initial_estimate is not None:
            self._check_solution_validity(
                initial_estimate, problem.Q.shape[0])
            mc_initial_solution = np.append(initial_estimate, 0)

        super().__init__(problem, params,
                         mc_initial_solution,
                         collect_heur_data)

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
