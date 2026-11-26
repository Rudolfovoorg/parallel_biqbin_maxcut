__version__ = "2.0.5"

from logging import INFO
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from biqbin.biqbin_module import (
    BabNode,
    Problem,
    abort_mpi,
    get_fixed_value,
    get_rank,
    get_root_sdp_bound,
    goemans_williamson_heuristic,
    reduce_sum_mpi,
    run,
    sdp_bound,
    set_heuristic,
    set_node_evaluation,
    set_primal_solution,
    update_mc_lower_bound_solution,
)
from biqbin.logging import logger
from biqbin.problem import ProblemBQP, ProblemMaxCut, ProblemQubo
from biqbin.solution import SolutionBQP, SolutionMaxCut, SolutionQubo
from biqbin.utils import PrettyPrint, data_collector


class MaxCutSolver(PrettyPrint):
    """Default MaxCut Biqbin Python Wrapper, solves the MaxCut problem."""

    solver_name: ClassVar[str] = f"PyBiqBin-MaxCut {__version__}"

    # Biqbin solution has 1 element less than the problem size, the last elemenent is assumed to be 0
    _MC_OFFSET: ClassVar[int] = 1

    def __init__(
        self,
        problem: ProblemMaxCut,
        params: str = "params",
        time_limit: int = 0,
        initial_estimate: np.ndarray | None = None,
        collect_heuristic_root_data: bool = False,
        collect_sdp_bound_root_data: bool = False,
    ):
        """Initialize the solver

        Args:
            problem (ProblemMaxCut): Problem instance to be solved
            params (str): path to parameters file
            time_limit (int): time limit in seconds
            initial_estimate (np.ndarray or None): Optional initial estimate solution binary vector
            collect_heuristic_root_data (bool): Collect heuristic runtimes and objective
                values on the root rank.
            collect_sdp_bound_root_data (bool): Collect SDP-bound runtimes and bounds
                on the root rank.
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
                    initial_estimate, problem.maxcut_adjacency_matrix.shape[0]
                )
                self.initial_estimate_solution = initial_estimate
                self.initial_heuristic = self._use_initial_estimate_on_root
                logger.info(f"Using an initial estimate solution: {self.initial_estimate_solution}")

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
        self.collect_heuristic_root_data: bool = (
            collect_heuristic_root_data if self.rank == 0 else False
        )
        self.heuristic_root_data: list[dict] = []
        self.sdp_bound_call_count = 0
        self.collect_sdp_bound_root_data: bool = (
            collect_sdp_bound_root_data if self.rank == 0 else False
        )
        self.sdp_bound_root_data: list[dict] = []

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
            self._overridden_sdp.append("sdp_bound")
        if type(self).initial_sdp_bound is not MaxCutSolver.initial_sdp_bound:
            self._overridden_sdp.append("initial_sdp_bound")

        if self.rank == 0:
            if self._overridden_sdp:
                logger.warning(
                    f"{type(self).__name__} overrides {' and '.join(self._overridden_sdp)}. "
                    f"Biqbin is an exact solver thus the returned value must be a valid upper bound "
                    f"on the subproblem optimal value. A bound that is too low will cause incorrect "
                    f"branch pruning and an invalid optimal solution. "
                )

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

    def initial_sdp_bound(self, node: BabNode, P0: Problem, P: Problem, *args, **kwargs) -> float:
        """Compute the SDP bound on the root node.
        By default it will call ``self.sdp_bound``, same as the leaf B&B nodes.

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
        P0: Problem = kwargs["P0"]
        node: BabNode = kwargs["node"]

        if not self._primal_solution_set:
            logger.fatal("SDP primal solution was not set before calling the default heuristc!")
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

        NOTE: The matrix must be in [-1, 1] range.

        Args:
            primal_solution (np.ndarray): shape (P.n, P.n) where n is the size of the subproblem P passed into
                ``heuristic`` and ``sdp_bound`` solver callbacks.
        """
        fatal_error = False
        if not primal_solution.ndim == 2:
            logger.fatal(
                f"primal_solution must be a 2D matrix, got {primal_solution.ndim}D matrix!"
            )
            fatal_error = True

        PRIMAL_SOLUTION_TOL = 1e-5
        if np.any(primal_solution < -1.0 - PRIMAL_SOLUTION_TOL) or np.any(
            primal_solution > 1.0 + PRIMAL_SOLUTION_TOL
        ):
            logger.fatal("The primal_solution must be in the [-1, 1] range!")
            fatal_error = True

        if not np.all(np.isfinite(primal_solution)):
            logger.fatal("The primal_solution must contain only finite values!")
            fatal_error = True

        if fatal_error:
            abort_mpi(10)

        set_primal_solution(primal_solution)
        self._primal_solution_set = True

    def _bab_node_evaluation(self, node: BabNode, P0: Problem, P: Problem) -> float:
        """Compute the upper and lower bound of the current B&B node

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
            if self._latest_heuristic_value > upper_bound_value + fixed_value:
                logger.fatal(
                    (
                        f"[rank {self.rank}] SDP bound is below the feasible heuristic value, "
                        f"custom SDP bound confirmed to not give a true bound!\n"
                        f"Heuristic value = {self._latest_heuristic_value}; SDP bound {upper_bound_value + fixed_value}"
                    )
                )
                fatal_error = True

            initial_sdp_bound = get_root_sdp_bound()
            if self.rank != 0 and self._latest_heuristic_value > initial_sdp_bound:
                logger.fatal(
                    (
                        f"[rank {self.rank}] Initial SDP bound is below the feasible heuristic value, "
                        f"custom SDP bound confirmed to not give a true bound!\n"
                        f"Heuristic value = {self._latest_heuristic_value}; initial SDP bound {initial_sdp_bound}"
                    )
                )
                fatal_error = True

            if fatal_error:
                abort_mpi(10)

        return upper_bound_value

    @data_collector(enabled_flag="collect_sdp_bound_root_data", data_box="sdp_bound_root_data")
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
                "SDP primal solution was not set before leaving `self.sdp_bound`!\n"
                "Please set the primal solution with `self.set_sdp_primal_solution` method!"
            )
            abort_mpi(10)
        if not np.isfinite(sdp_bound_value):
            logger.fatal("sdp_bound must return a finite float")
            abort_mpi(10)
        logger.debug(
            f"SDP bound value = {sdp_bound_value}; B&B node level = {node.level}; B&B parent SDP bound value {node.upper_bound}"
        )
        return sdp_bound_value

    @data_collector(enabled_flag="collect_heuristic_root_data", data_box="heuristic_root_data")
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
        self._check_solution_validity(heur_sol, P.n - MaxCutSolver._MC_OFFSET)

        # copy to full solution x
        x = node.sol.x.copy()
        x[node.xfixed == 0] = heur_sol

        self._latest_heuristic_value = self._evaluate_solution(P0.L, x)
        update_mc_lower_bound_solution(x)

        logger.debug("Heuristic value: %s", self._latest_heuristic_value)

        return self._latest_heuristic_value

    def _use_initial_estimate_on_root(self, L: np.ndarray, *args, **kwargs) -> npt.ArrayLike:
        """Heuristic call on root node if initial estimate solution is used"""
        fatal_error = False
        if kwargs["P0"].L.shape != L.shape:
            logger.fatal(
                f"Original problem shape {kwargs['P0'].L.shape} != Subproblem shape {L.shape}!"
            )
            fatal_error = True
        if kwargs["P"].L.shape != self.problem.maxcut_adjacency_matrix.shape:
            logger.fatal(
                f"Original problem shape {kwargs['P0'].L.shape} != mc adjacency matrix shape {self.problem.maxcut_adjacency_matrix.shape}!"
            )
            fatal_error = True
        if np.any(kwargs["node"].xfixed):
            logger.fatal("xfixed is nonzero!")
            fatal_error = True

        if fatal_error:
            abort_mpi(10)

        assert self.initial_estimate_solution is not None
        return self.initial_estimate_solution[: -MaxCutSolver._MC_OFFSET]

    def _evaluate_solution(self, L0: np.ndarray, sol: np.ndarray) -> float:
        """Calculate the Max-Cut objective value of the solution

        Args:
            L0 (np.ndarray): Laplacean objective matrix of the original (full) problem
            sol (np.ndarray): current solution

        Returns:
            float: value of the solution
        """
        self._check_solution_validity(
            sol, self.problem.maxcut_adjacency_matrix.shape[0] - MaxCutSolver._MC_OFFSET
        )
        sol_val = sol @ L0[: -MaxCutSolver._MC_OFFSET, : -MaxCutSolver._MC_OFFSET] @ sol

        return float(sol_val)

    def _run_solver(self) -> dict | None:
        """Runs Biqbin C/C++ implementation

        Returns:
            dict | None: solution on MPI rank == 0; else None
        """
        # MC adjacency matrix is passed in on rank 0 and broadcasted to worker ranks during execution
        if self.rank == 0:
            input_matrix = self.problem.maxcut_adjacency_matrix.astype(np.float64)
            if logger.isEnabledFor(INFO):  # lazy evaluating
                logger.info("%s", self.problem.format_summary(verbose=True))
        else:
            input_matrix = None

        # Native callbacks are process-global. Bind the solver being executed,
        # rather than whichever solver object happened to be constructed last.
        set_node_evaluation(self._bab_node_evaluation)
        set_heuristic(self._call_heuristic)
        raw_results = run(
            self.solver_name, self.problem.problem_name, input_matrix, self.params, self.time_limit
        )

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
        bab_node_call_count_sum = reduce_sum_mpi(self.bab_node_evaluation_call_count)
        sdp_call_count_sum = reduce_sum_mpi(self.sdp_bound_call_count)
        heur_call_count_sum = reduce_sum_mpi(self.heuristic_call_count)

        if self.rank != 0:
            return None

        raw_result["meta_data"]["solver"] = self.solver_name
        raw_result["meta_data"]["instance"] = self.problem.problem_name
        raw_result["meta_data"]["parameters"] = {
            "time_limit": self.time_limit if self.time_limit > 0 else None,
            "optimized": self.problem.optimize_mc_adj_matrix,
            "gcd": self.problem.gcd,
        }
        if self.initial_estimate_solution is not None:
            raw_result["meta_data"]["initial_estimate_solution"] = (
                self.initial_estimate_solution.tolist()
            )

        raw_result["meta_data"]["bab_node_evaluation_call_count"] = bab_node_call_count_sum
        raw_result["meta_data"]["sdp_call_count"] = sdp_call_count_sum
        raw_result["meta_data"]["heuristic_call_count"] = heur_call_count_sum

        if self.collect_heuristic_root_data:
            raw_result["meta_data"]["root_node"]["total_heur_time"] = sum(
                d["time"] for d in self.heuristic_root_data
            )
            raw_result["meta_data"]["root_node"]["heuristic_call_count"] = self.heuristic_call_count
            raw_result["meta_data"]["root_node"]["heuristic_data"] = self.heuristic_root_data
        if self.collect_sdp_bound_root_data:
            raw_result["meta_data"]["root_node"]["sdp_bound_data"] = self.sdp_bound_root_data
        return raw_result

    def _check_solution_validity(self, solution: np.ndarray, problem_size: int):
        """Check if the passed in solution is in correct format (binary 0-1 vector) for Biqbin solver."""
        fatal_error = False
        error_msg = ""
        if solution.ndim != 1 or solution.shape[0] != problem_size:
            error_msg += f"Solution vector must be a 1D vector of size {problem_size}, but got shape {solution.shape}! "
            fatal_error = True
        if not ((solution == 0) | (solution == 1)).all():
            error_msg += "Solution vector must be a binary [0, 1] vector!"
            fatal_error = True

        if fatal_error:
            logger.fatal(error_msg, stack_info=True)
            abort_mpi(10)

    def __str__(self) -> str:
        return (
            f"{super().__str__()}"
            f"    Problem = {self.problem.problem_name}\n"
            f"Params path = {self.params}\n"
            f" Time limit = {self.time_limit}\n"
        )


class QUBOSolver(MaxCutSolver):
    """Solve a QUBO through its Max-Cut reformulation.

    Call ``biqbin.init()`` first and call ``compute()`` on every MPI rank.
    Constructor options follow ``MaxCutSolver``; ``initial_estimate`` is a
    binary vector with one entry per original QUBO variable.
    """

    solver_name = f"PyBiqBin-QUBO {__version__}"

    def __init__(
        self,
        problem: ProblemQubo,
        params: str = "params",
        time_limit: int = 0,
        initial_estimate: np.ndarray | None = None,
        collect_heuristic_root_data: bool = False,
        collect_sdp_bound_root_data: bool = False,
    ):

        mc_initial_solution = None
        if get_rank() == 0 and initial_estimate is not None:
            self._check_solution_validity(initial_estimate, problem.Q.shape[0])
            # Maxcut solution is one larger than Qubo solution
            mc_initial_solution = np.append(initial_estimate, 0)

        super().__init__(
            problem,
            params,
            time_limit,
            mc_initial_solution,
            collect_heuristic_root_data,
            collect_sdp_bound_root_data,
        )

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


class BQPSolver(MaxCutSolver):
    """Solve an equality-constrained binary quadratic minimization problem.

    Call ``biqbin.init()`` before constructing the solver. Constructor options
    follow ``MaxCutSolver``; an ``initial_estimate`` must satisfy ``A @ x == b``.
    Every MPI rank must participate in ``compute()``.
    """

    solver_name = "PyBiqBin-BQP"

    def __init__(
        self,
        problem: ProblemBQP,
        params: str,
        time_limit: int = 0,
        initial_estimate=None,
        collect_heuristic_root_data=False,
        collect_sdp_bound_root_data=False,
    ):
        self.__problem: ProblemBQP = problem

        if initial_estimate is not None:
            # Transform BQP solution to Biqbin Max-Cut solution vector, throwing error if not feasible
            initial_estimate = self._initial_estimate_bqp_to_maxcut(initial_estimate)

        super().__init__(
            problem=problem,
            params=params,
            time_limit=time_limit,
            initial_estimate=initial_estimate,
            collect_heuristic_root_data=collect_heuristic_root_data,
            collect_sdp_bound_root_data=collect_sdp_bound_root_data,
        )

    @property
    def problem(self) -> ProblemBQP:
        return self.__problem

    def compute(self) -> SolutionBQP | None:
        """Run the solver collectively and recover the original BQP result.

        Returns:
            SolutionBQP | None: Result on MPI rank 0; ``None`` on workers.
                Check ``solution["feasible_solution"]`` before using ``x``.
        """
        result = self._run_solver()
        if result is not None:
            return SolutionBQP(result, self.problem)

    def _initial_estimate_bqp_to_maxcut(self, bqp_x):
        """Transforms the BQP solution vector to a Max-Cut vector.

        Args:
            bqp_x (np.ndarray): BQP binary vector solution

        Raises:
            ValueError: bqp_x must be a feasable solution to the problem

        Returns:
            np.ndarray: Max-Cut solution vector
        """
        bqp_x = np.asarray(bqp_x)
        n = self.__problem.F.shape[0]

        if (
            bqp_x.ndim != 1
            or bqp_x.shape != (n,)
            or not np.all(np.isfinite(bqp_x))
            or not np.all((bqp_x == 0) | (bqp_x == 1))
        ):
            logger.fatal(f"Initial estimate must be a binary vector of length {n}.")
            abort_mpi(10)

        if not np.array_equal(
            self.__problem.A @ bqp_x,
            self.__problem.b,
        ):
            logger.fatal("Initial estimate does not satisfy A @ x = b.")
            abort_mpi(10)

        bqp_obj = bqp_x @ self.__problem.F @ bqp_x + self.__problem.c @ bqp_x

        for candidate in (bqp_x, 1 - bqp_x):
            # append the fixed auxiliary coordinate
            mc_candidate = np.append(candidate, 0)
            y = 1 - 2 * mc_candidate  # 0/1 partition -> {-1, 1}
            cut_value = 0.25 * (
                self.__problem.maxcut_adjacency_matrix.sum()
                - y @ self.__problem.maxcut_adjacency_matrix @ y
            )

            cut_value *= self.__problem.gcd  # undo Max-cut adjacency matrix optimization
            if np.isclose(self.__problem.const_value - cut_value, bqp_obj):
                return mc_candidate

        logger.fatal(
            "initial estimate solution is not a feasible solution to the problem.\n"
            "Please input a feasible initial estimate or run without passing one in.",
            stack_info=True,
        )
        abort_mpi(10)
