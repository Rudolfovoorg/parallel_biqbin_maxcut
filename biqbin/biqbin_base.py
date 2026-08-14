__version__ = '2.0.5'

from typing import ClassVar

import numpy.typing as npt
import numpy as np
import logging


from biqbin.utils import (check_matrix_validity_setter_wrap,
                          check_matrix_validity_wrap,
                          divide_matrix_by_gcd, 
                          data_collector)
from biqbin.biqbin_module import (BabNode, Problem,
                                  reduce_sum_mpi, abort_mpi, get_rank, run,
                                  update_mc_lower_bound_solution,
                                  set_heuristic, goemans_williamson_heuristic,
                                  set_primal_solution, set_node_evaluation, sdp_bound,
                                  get_fixed_value, get_root_sdp_bound)

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
    def maxcut_adjacency_matrix(self) -> npt.NDArray[np.floating | np.integer]:
        """Returns the valid input which Biqbin can solve.
        Checks if the input is valid for Biqbin (if all values are integers).

        Returns:
            np.ndarray: adjacency matrix for the MaxCut problem
        """

        return self._maxcut_adjacency_matrix

    @maxcut_adjacency_matrix.setter
    @check_matrix_validity_setter_wrap
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

    @check_matrix_validity_setter_wrap
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
                    f'Heurist run count     = {self.meta_data['heuristic_call_count']}\n'
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
    _MC_OFFSET: ClassVar[int] = 1

    def __init__(self,
                 problem: ProblemMaxCut,
                 params: str = 'params',
                 time_limit: int = 0,
                 initial_estimate: np.ndarray | None = None,
                 collect_heuristic_root_data: bool = False,
                 collect_sdp_bound_root_data: bool = False):
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
                self.initial_heuristic = self._use_initial_estimate_on_root
                logger.info(
                    f'Using an initial estimate solution: {self.initial_estimate_solution}')

            self._heuristic_fn = self.initial_heuristic
            self._sdp_bound_fn = self.initial_sdp_bound

        # On worker rank
        else:
            # Use the non-root heuristic and upper_bound methods on worker processes
            self._heuristic_fn = self.heuristic
            self._sdp_bound_fn = self.sdp_bound

        # Data collection
        # BZ TODO: Enable non-root data collection
        self.bab_node_evaluation_call_count = 0
        self.heuristic_call_count = 0
        self.collect_heuristic_root_data: bool = collect_heuristic_root_data if self.rank == 0 else False
        self.heuristic_root_data: list[dict] = []
        self.sdp_bound_call_count = 0
        self.collect_sdp_bound_root_data: bool = collect_sdp_bound_root_data if self.rank == 0 else False
        self.sdp_bound_root_data: list[dict] = []

        # Set the functions in C++ source to be called
        set_node_evaluation(self._bab_node_evaluation)
        set_heuristic(self._call_heuristic)

        # biqbin_module.sdp_bound needs to be called before goemans_williams_heuristic is called
        self._primal_solution_set: bool = False
        # SDPBound calls heuristic by itself, if that is overwritten, we need to call it manually
        self._heuristic_was_called: bool = False
        # if the sdp bound is overridden we need to compare the current nodes sdp value with
        # the sdp bound to assure optimality
        self._latest_heuristic_value: float = 0

        # Warnings about the validity of the solution
        self._overridden_sdp = []
        if type(self).sdp_bound is not MaxCutSolver.sdp_bound:
            self._overridden_sdp.append('sdp_bound')
        if type(self).initial_sdp_bound is not MaxCutSolver.initial_sdp_bound:
            self._overridden_sdp.append('initial_sdp_bound')

        if self.rank == 0:
            if self._overridden_sdp:
                logger.warning(
                    f'{type(self).__name__} overrides {" and ".join(self._overridden_sdp)}. '
                    f'Biqbin is an exact solver thus the returned value must be a valid upper bound '
                    f'on the subproblem optimal value. A bound that is too low will cause incorrect '
                    f'branch pruning and an invalid optimal solution. ')

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

    def sdp_bound(self, node: BabNode, P0: Problem, P: Problem, *args, **kwargs) -> float:
        """Compute the SDP bound for a B&B node.

        Override in subclasses to use a custom bounding strategy.
        """
        self._primal_solution_set = True
        return sdp_bound(node, P0, P)

    # TODO: throw warning about if it is a true lower bound, throw error if bellow heuristic solution
    def initial_sdp_bound(self, node: BabNode, P0: Problem, P: Problem, *args, **kwargs) -> float:
        """Compute the SDP bound on the root node. 
        By default it will call ``self.upper_bound``, same as the leaf B&B nodes.

        Override in subclasses to use a custom bounding strategy.
        """
        return self.sdp_bound(node, P0, P, *args, **kwargs)

    def heuristic(self, L: np.ndarray, *args, **kwargs) -> npt.ArrayLike:
        """Finds a heuristic solution binary vector for the subproblem Laplacean matrix L.

        Args:
            L (np.ndarray): Subproblem objective matrix for the Max-Cut Problem

        Keyword Args:
            node (BabNode): current B&B node
            P0 (Problem): Original (full) problem
            P (Problem): Current nodes subproblem

        Returns:
            npt.ArrayLike: Binary solution vector found by the heuristic. 
            Shape must be a 1D of length one less than subproblem size (L.shape[0] - 1, )
        """
        P0: Problem = kwargs['P0']
        node: BabNode = kwargs['node']

        if not self._primal_solution_set:
            logger.fatal(
                f'SDP primal solution was not set before calling the default heuristc!')
            abort_mpi(10)

        # biqbin GW implementation expects a full solution vector
        x = np.zeros(P0.n - MaxCutSolver._MC_OFFSET, dtype=np.int32)

        # solution is stored in x where solution variables are not fixed
        goemans_williamson_heuristic(P0.L, L, node.xfixed, node.sol.x, x)

        # return only where solution variables are not fixed
        return x[node.xfixed == 0]

    def initial_heuristic(self, L: np.ndarray, *args, **kwargs) -> npt.ArrayLike:
        """Finds a heuristic solution binary vector on the root B&B node.
        By default it calls ``self.heuristic``, same as the leaf B&B nodes.

        Args:
            L (np.ndarray): Subproblem objective matrix for the Max-Cut Problem

        Keyword Args:
            node (BabNode): current B&B node
            P0 (Problem): Original (full) problem
            P (Problem): Current nodes subproblem

        Returns:
            npt.ArrayLike: Binary solution vector found by the heuristic.
            Shape must be a 1D of length one less than full problem size (P0.n - 1, )
        """
        return self.heuristic(L, *args, **kwargs)

    def set_sdp_primal_solution(self, primal_solution: npt.NDArray[np.float64]):
        """Sets the SDP primal solution matrix X.

        Vital for branching and the GW heuristic, default SDPBound routine in C does this automatically, 
        but if we overwrite ``sdp_bound`` with a custom implementation, we need to manually set the primal solution X before running.

        NOTE: The matrix should be in {-1, 1} range.

        Args:
            primal_solution (np.ndarray): shape (P.n, P.n) where n is the size of the subproblem P passed into
            ``heuristic`` and ``sdp_bound`` solver callbacks.
        """
        if np.any(primal_solution < -1.0) or np.any(primal_solution > 1.0):
            logger.fatal(
                'The primal_solution must be in the {-1, 1} range!')
            abort_mpi(10)

        if not np.all(np.isfinite(primal_solution)):
            logger.fatal(
                "The primal_solution must contain only finite values!")
            abort_mpi(10)

        set_primal_solution(primal_solution)
        self._primal_solution_set = True

    def _bab_node_evaluation(self, node: BabNode, P0: Problem, P: Problem) -> float:
        """ Compute the upper and lower bound of the current B&B node

        Args:
            node (BabNode): current B&B node
            P0 (Problem): original (full) problem
            P (Problem): current nodes subproblem

        Returns:
            float: sdp bound value of the current node
        """
        self.bab_node_evaluation_call_count += 1

        # Check if the primal solution was set for the default GW heuristic,
        # default sdp_bound does this by itself.
        self._primal_solution_set = False

        # Biqbin's default sdp_bound calls heuristic function
        # but if we use a different sdp functions we need to call it manually
        self._heuristic_was_called = False

        upper_bound_value = self._call_sdp_bound(node, P0, P)

        if not self._heuristic_was_called:
            self._call_heuristic(node, P0, P)

        if self._overridden_sdp:
            # If any of the sdp bound computations were overridden,
            # we need to do a runtime check to see if the computed bounds are valid
            fatal_error = False
            
            fixed_value = get_fixed_value(node, P0)
            if (self._latest_heuristic_value > upper_bound_value + fixed_value):
                logger.fatal((f"[rank {self.rank}] SDP bound is greater than heuristic bound, "
                              f"custom SDP bound confirmed to not give a true bound!\n"
                              f"Heuristic value = {self._latest_heuristic_value}; SDP bound {upper_bound_value + fixed_value}"))
                fatal_error = True
                
            initial_sdp_bound = get_root_sdp_bound()
            if (self.rank != 0 and self._latest_heuristic_value > initial_sdp_bound):
                logger.fatal((f"[rank {self.rank}] Initial SDP bound is greater than heuristic bound, "
                              f"custom SDP bound confirmed to not give a true bound!\n"
                              f"Heuristic value = {self._latest_heuristic_value}; initial SDP bound {initial_sdp_bound}"))
                fatal_error = True

            if fatal_error:
                abort_mpi(10)

        return upper_bound_value

    @data_collector(enabled_flag='collect_sdp_bound_root_data', data_box='sdp_bound_root_data')
    def _call_sdp_bound(self, node: BabNode, P0: Problem, P: Problem) -> float:
        """Wrapper around sdp bound function call.

        Contains validity checks and optional runtime data.
        This is the function native Biqbin calls from C.

        Returns:
            float: SDP value
        """
        self.sdp_bound_call_count += 1
        sdp_bound_value = self._sdp_bound_fn(node, P0, P)

        if not self._primal_solution_set:
            logger.fatal(
                f'SDP primal solution was not set before leaving `self.sdp_bound`!\n'
                f'Please set the primal solution with `self.set_primal_solution` method!')
            abort_mpi(10)
        if not np.isfinite(sdp_bound_value):
            logger.fatal("sdp_bound must return a finite float")
            abort_mpi(10)

        return sdp_bound_value

    @data_collector(enabled_flag='collect_heuristic_root_data', data_box='heuristic_root_data')
    def _call_heuristic(self, node: BabNode, P0: Problem, P: Problem) -> float:
        """Wrapper around heuristic function, it validates the provided solution vector by ``self.heuristic``
        before sending it to the native solver. It also collects additional metadata (runtime and return value).

        This is the function native Biqbin calls from C.

        Args:
            node (BabNode): current B&B node
            P0 (Problem): original (full) problem
            P (Problem): current nodes subproblem

        Returns:
            float: objective value of the heuristic solution
        """
        # Call heuristic function
        self.heuristic_call_count += 1
        heur_sol = self._heuristic_fn(P.L, node=node, P0=P0, P=P)
        self._heuristic_was_called = True

        # Check if the solution is in valid format
        if not isinstance(heur_sol, np.ndarray):
            heur_sol = np.array(heur_sol)
        self._check_solution_validity(
            heur_sol, P.n - MaxCutSolver._MC_OFFSET)

        # copy to full solution x
        x = node.sol.x.copy()
        x[node.xfixed == 0] = heur_sol

        self._latest_heuristic_value = self._evaluate_solution(P0.L, x)
        update_mc_lower_bound_solution(x)

        if logger.isEnabledFor(logging.DEBUG):
            if self._primal_solution_set:
                default_gw_value = goemans_williamson_heuristic(
                    P0.L, P.L, node.xfixed, node.sol.x, np.zeros(P0.n - 1)
                )
                logger.debug(
                    f'Custom heuristic: {self._latest_heuristic_value}; default gw heuristic: {default_gw_value}'
                )

        return self._latest_heuristic_value

    def _use_initial_estimate_on_root(self, L: np.ndarray, *args, **kwargs) -> npt.ArrayLike:
        """ Heuristic call on root node if initial estimate solution is used
        """
        fatal_error = False
        if kwargs['P0'].L.shape != L.shape:
            logger.fatal(
                f"Original problem shape {kwargs['P0'].L.shape} != Subproblem shape {L.shape}!")
            fatal_error = True
        if kwargs['P'].L.shape != self.problem.maxcut_adjacency_matrix.shape:
            logger.fatal(
                f"Original problem shape {kwargs['P0'].L.shape} != mc adjacency matrix shape {self.problem.maxcut_adjacency_matrix.shape}!")
            fatal_error = True
        if np.any(kwargs['node'].xfixed):
            logger.fatal("xfixed is nonzero!")
            fatal_error = True

        if fatal_error:
            abort_mpi(10)

        assert self.initial_estimate_solution is not None
        return self.initial_estimate_solution[:-MaxCutSolver._MC_OFFSET]

    def _evaluate_solution(self, L0: np.ndarray, sol: np.ndarray) -> float:
        """Calculate the Max-Cut objective value of the solution

        Args:
            L0 (np.ndarray): Laplacean objective matrix of the original (full) problem
            sol (np.ndarray): current solution

        Returns:
            float: value of the solution
        """
        self._check_solution_validity(
            sol, self.problem.maxcut_adjacency_matrix.shape[0] - MaxCutSolver._MC_OFFSET)
        sol_val = sol @ L0[:-MaxCutSolver._MC_OFFSET,
                           :-MaxCutSolver._MC_OFFSET] @ sol

        return float(sol_val)

    def _run_solver(self) -> dict | None:
        """Runs Biqbin C/C++ implementation

        Returns:
            dict | None: solution on MPI rank == 0; else None
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

        result = self._update_result_dict(raw_results)
        return result

    def _update_result_dict(self, raw_result: dict) -> dict | None:
        """Update the raw result from BiqBin C implementation with additional metadata.
        Calls ``MPI_Reduce`` to sum up data from worker ranks. 

        Args:
            raw_result (dict): result dictionary built in C-solver, is empty on ``self.rank != 0``

        Returns:
            dict | None: updated result on ``self.rank == 0``; else None
        """
        bab_node_call_count_sum = reduce_sum_mpi(
            self.bab_node_evaluation_call_count)
        sdp_call_count_sum = reduce_sum_mpi(self.sdp_bound_call_count)
        heur_call_count_sum = reduce_sum_mpi(self.heuristic_call_count)

        if self.rank != 0:
            return None

        raw_result['meta_data']['solver'] = self.solver_name
        raw_result['meta_data']['instance'] = self.problem.problem_name
        raw_result['meta_data']['parameters'] = {
            'time_limit': self.time_limit if self.time_limit > 0 else None,
            'optimized': self.problem.optimize_mc_adj_matrix,
            'gcd': self.problem.gcd
        }
        if self.initial_estimate_solution is not None:
            raw_result['meta_data']['initial_estimate_solution'] = self.initial_estimate_solution.tolist()

        raw_result['meta_data']['bab_node_evaluation_call_count'] = bab_node_call_count_sum
        raw_result['meta_data']['sdp_call_count'] = sdp_call_count_sum
        raw_result['meta_data']['heuristic_call_count'] = heur_call_count_sum

        if self.collect_heuristic_root_data:
            raw_result['meta_data']['root_node']['total_heur_time'] = sum(
                d['time'] for d in self.heuristic_root_data)
            raw_result['meta_data']['root_node']['heuristic_call_count'] = self.heuristic_call_count
            raw_result['meta_data']['root_node']['heuristic_data'] = self.heuristic_root_data
        if self.collect_sdp_bound_root_data:
            raw_result['meta_data']['root_node']['sdp_bound_data'] = self.sdp_bound_root_data
        return raw_result

    def _check_solution_validity(self, solution: np.ndarray, problem_size: int):
        """Check if the passed in solution is in correct format (binary 0-1 vector) for Biqbin solver."""
        fatal_error = False
        error_msg = ''
        if solution.ndim != 1 or solution.shape[0] != problem_size:
            error_msg += f'Solution vector must be a 1D vector of size {problem_size}, but got shape {solution.shape}! '
            fatal_error = True
        if not ((solution == 0) | (solution == 1)).all():
            error_msg += 'Solution vector must be a binary [0, 1] vector!'
            fatal_error = True

        if fatal_error:
            logger.fatal(error_msg, stack_info=True)
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
                 collect_heuristic_root_data: bool = False,
                 collect_sdp_bound_root_data: bool = False
                 ):

        mc_initial_solution = None
        if get_rank() == 0 and initial_estimate is not None:
            self._check_solution_validity(
                initial_estimate, problem.Q.shape[0])
            # Maxcut solution is one larger than Qubo solution
            mc_initial_solution = np.append(initial_estimate, 0)

        super().__init__(problem, params, time_limit,
                         mc_initial_solution,
                         collect_heuristic_root_data,
                         collect_sdp_bound_root_data)

        self.__problem: ProblemQubo = problem

    @property
    def problem(self) -> ProblemQubo:
        return self.__problem

    def compute(self) -> SolutionQubo | None:
        """Computes the solution to the QUBO using Biqbin, only MPI rank == 0 returns Solution

        Returns:
            SolutionQubo: solution class for QUBO problem. returns None if MPI rank != 0.
        """
        biqbin_result = self._run_solver()

        if biqbin_result is not None:
            return SolutionQubo(biqbin_result, self.problem)
        else:
            return None
