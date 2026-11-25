# biqbin.pyi
import numpy as np
import numpy.typing as npt
from typing import Callable, Tuple


"""
Python bindings for BiqBin exposed in biqbin_module.so
"""


def run(solver_name: str, problem_instance_path: str, maxcut_adj_matrix: npt.NDArray[np.float64] | None, params_path: str, time_limit: int) -> dict:
    """Runs the solver, returns a solution dictionary on mpi rank 0"""
    ...


def set_heuristic(heuristic_function: Callable[[BabNode, Problem, Problem], float]) -> None:
    """Sets the heuristic function in native biqbin."""
    ...


def set_primal_solution(primal_solution: npt.NDArray[np.float64]) -> None:
    """Sets the SDP primal solution matrix ``X`` in C.

    Vital step before running the default GW heuristic or branching,
    default SDPBound routine does this automatically, but if we overwrite ``sdp_bound`` with a 
    custom implementation, we need to manually set the primal solution X

    Args:
        primal_solution (np.ndarray): shape (P.n, P.n) where n is the size of the subproblem P passed into
        ``heuristic`` and ``sdp_bound`` solver callbacks. It is in {-1, 1} range.
    """
    ...


def get_fixed_value(node: BabNode, P0: Problem) -> float:
    """Calculate the objective value contribution of the "fixed" part,
    where ``node.xfixed[i] == 1``.

    Args:
        node (BabNode): Current B&B node
        P0 (Problem): Main (full) Problem

    Returns:
        float: Fixed value
    """
    ...


def update_mc_lower_bound_solution(new_solution_x: npt.NDArray[np.int32]) -> bool:
    """Update Max-Cut global lower-bound solution, if it is better than the current one

    Args:
        new_solution_x (npt.NDArray[np.int32]): Binary vector of a potential new solution.

    Returns:
        bool: True if solution is updated (new_solution_x is better than the current one), False otherwise
    """
    ...


def goemans_williamson_heuristic(L0: np.ndarray, L: np.ndarray, xfixed: np.ndarray, sol_X: np.ndarray, x: np.ndarray) -> float:
    """Default Biqbin GW heuristic implementation"""
    ...


def set_node_evaluation(node_eval_function: Callable[[BabNode, Problem, Problem], float]) -> None:
    """Set the function that will evaluate upper and lower bounds of a branch-and-bound node.
    """
    ...


def sdp_bound(node: BabNode, P0: Problem, P: Problem) -> float:
    """Default Biqbin SDPBound implementation which internally calls the heuristic function many times during execution.

    Args:
        node (BabNode): current node being evaluated
        P0 (Problem): Full problem, constructed at the start of solving
        P (Problem): Subproblem constructed for the current node

    Returns:
        float: SDP relaxation value
    """
    ...


def get_root_sdp_bound() -> float:
    """Get the sdp bound value computed on the root node,
    used to check the validity of a custom sdp bound.

    Returns:
        float: sdp bound value saved in the native solver as double
    """


def get_rank() -> int:
    """Returns the MPI rank."""
    ...


def init_mpi() -> Tuple[int, int]:
    """initializes MPI

    Returns:
        (int, int): MPI (size, rank) tuple 
    """
    ...


def finalize_mpi() -> None:
    """Finalize MPI protocol after running the solver
    """


def abort_mpi(abort_code: int):
    """Abort solver execution on fatal errors

    Args:
        abort_code (int)
    """
    ...


def reduce_sum_mpi(number: int | float) -> int | float:
    """Reduce sum the input number of all ranks

    Returns:
        int | float: the sum on master process ``MPI rank == 0``, else it returns 0. 
    """
    ...


def interior_point_method_maxcut(L: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Solves the basic SDP relaxation of Max-Cut using the IPM_MC_PK
    primal-dual predictor-corrector interior-point method.

    Primal:
        maximize    np.trace(L @ X)
        subject to  np.diag(X) == np.ones(n)
                    X is positive semidefinite

    Dual:
        minimize    np.ones(n) @ y
        subject to  np.diag(y) - L is positive semidefinite
                    y is unrestricted

    Args:
        L (np.ndarray): Symmetric objective PSD matrix.

    Returns:
        tuple[float, np.ndarray]:
            Optimal SDP value, given by the dual objective, and the
            corresponding optimal primal matrix X.
    """
    ...


class BabSolution:
    """Solution data attached to a branch-and-bound node.

    Python view of the native ``BabSolution`` struct used by the solver.

    The object is accessed through ``BabNode.sol`` during custom
    node-evaluation callbacks.
    """
    @property
    def x(self) -> npt.NDArray[np.int32]:
        """Binary 0-1 solution vector.

        Python view of the native ``BabSolution.X`` array.

        One-dimensional read-only Python view with shape ``(P0.n - 1,)``, where
        ``P0`` is the full problem passed to the same callback.

        For entries where ``node.xfixed[i] == 1``, ``x[i]`` gives the binary value
        that variable is fixed to.

        The returned array is backed by solver-owned memory. Copy it before
        storing it beyond the current callback.
        """
        ...


class BabNode:
    """Branch-and-bound search node.

    Python view of the native ``BabNode`` struct used by the solver.

    A ``BabNode`` is passed to node-evaluation callbacks such as ``sdp_bound`` and ``heuristic``.

    Treat array-valued attributes as short-lived views. Copy them before storing
    them beyond the current callback.    
    """

    @property
    def xfixed(self) -> npt.NDArray[np.int32]:
        """Mask of problem variables fixed at this branch-and-bound node.

        One-dimensional read-only Python view with shape ``(P0.n - 1,)``,
        where ``P0`` is the main problem passed to the same callback.

        Entry ``xfixed[i]`` indicates
        whether the corresponding problem variable / graph vertex has already
        been fixed by branching:

        - ``0``: variable is still free at this search-tree node
        - ``1``: variable has been fixed at this search-tree node, fixed solutions
        value is stored in the ``sol.x[i]`` 

        This is a fixing mask, not the fixed value itself. The returned array
        is read-only and backed by solver-owned memory.
        """
        ...

    @property
    def sol(self) -> BabSolution:
        """Solution data associated with this search-tree node.

        Provides access to the native ``BabSolution`` object stored in the
        branch-and-bound node.

        ``xfixed`` mask indicates which problem variables have been fixed by branching.
        For entries where ``xfixed[i] == 1``, ``sol.x[i]`` gives the binary value
        that variable is fixed to.

        The returned object is a Python view of solver-owned state. Its array
        attributes are valid only while the current callback is active; copy any
        data that must be stored beyond the callback.
        """
        ...

    @property
    def fracsol(self) -> npt.NDArray[np.float64]:
        """Fractional solution associated with this node, used for determining the next branching variable.

        Shape: ``(P0.n - 1,)`` where n is the size of the main problem P0

        This array is intended for node-evaluation logic and may be updated by the solver between
        callbacks.

        The returned array is backed by solver-owned memory. Copy it before
        storing it beyond the current callback.
        """
        ...

    @property
    def level(self) -> int:
        """Depth of this node in the branch-and-bound tree.

        The root node has level ``0``.
        """
        ...

    @property
    def upper_bound(self) -> float:
        """Current upper bound associated with this node. It is set after node evaluation
        by the solver.
        """
        ...


class Problem:
    """Problem data passed to node-evaluation callbacks.

    Python view of the native ``Problem`` struct used by the solver.

    ``Problem`` represents either the full MaxCut problem or the current
    branch-and-bound subproblem, depending on which callback argument is being
    inspected. In callbacks such as ``sdp_bound(node, P0, P)``, ``P0`` is the
    full problem and ``P`` is the current subproblem.

    Array-valued attributes expose solver-owned memory and are read-only
    Python views. Copy arrays before storing them beyond the current callback.
    """
    @property
    def L(self) -> npt.NDArray[np.float64]:
        """Objective matrix.

        Two-dimensional read-only Python view with shape ``(self.n, self.n)``.

        - For ``P0``, this is the objective matrix of the full problem. 
        - For ``P``, this is the objective matrix of the current subproblem.

        The returned array is backed by solver-owned memory. Copy it before
        storing it beyond the current callback.
        """
        ...

    @property
    def n(self) -> int:
        """Size of the objective matrix ``L``.

        ``L`` has shape ``(n, n)``. For MaxCut problems this size includes the
        solver's dummy MaxCut variable, so the number of non-dummy problem
        variables is usually ``n - 1``.
        """
        ...

    @property
    def NIneq(self) -> int:
        """Number of triangle inequalities currently attached to this problem."""
        ...

    @property
    def NPentIneq(self) -> int:
        """Number of pentagonal inequalities currently attached to this problem."""
        ...

    @property
    def NHeptaIneq(self) -> int:
        """Number of heptagonal inequalities currently attached to this problem."""
        ...

    @property
    def bundle(self) -> int:
        """Bundle size used by the SDP bounding routine for this problem."""
        ...
