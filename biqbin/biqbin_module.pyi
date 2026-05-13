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


def set_heuristic(heuristic_function: Callable[[BabNode, Problem, Problem], float]):
    """Sets the heuristic function in biqbin."""
    ...


def set_primal_solution(primal_solution: np.ndarray):
    """Set the primal solution matrix X before running the default GW heuristic

    Args:
        primal_solution (np.ndarray): shape (n, n) where n is the size of the original Problem->n
    """
    ...


def update_mc_lower_bound_solution(new_solution_x: npt.NDArray[np.int32]) -> bool:
    """update the Max-Cut global lower-bound solution, if it is better than the current one

    Args:
        new_solution_x (npt.NDArray[np.int32]): Binary vector of a potential new solution.

    Returns:
        bool: True if solution is updated (new_solution_x is better than the current one), False otherwise
    """
    ...


def goemans_williamson_heuristic(L0: np.ndarray, L: np.ndarray, xfixed: np.ndarray, sol_X: np.ndarray, x: np.ndarray) -> float:
    """Default Biqbin GW heuristic implementation (heuristic_unpacked in src/heuristic.c)"""
    ...


def set_node_evaluation(node_eval_function: Callable[[BabNode, Problem, Problem], float]):
    """Set the function that will evaluate both bounds.
    """
    ...


def sdp_bound(node: BabNode, main_problem: Problem, subproblem: Problem) -> float:
    """Default Biqbin SDPBound in src/bounding.c. Calls the heuristic function many times during execution

    Args:
        node (BabNode): current node being evaluated
        main_problem (Problem): Full problem, constructed at the start of solving
        subproblem (Problem): Subproblem constructed for the current node

    Returns:
        float: Max-Cut upper bound (QUBO minimization lower bound)
    """
    ...


def get_rank() -> int:
    """Returns the MPI rank."""
    ...


def init_mpi() -> Tuple[int, int]:
    """initializes MPI

    Returns:
        (int, int): MPI (size, rank) tuple 
    """
    ...


def abort_mpi(abort_code: int):
    """Abort solver execution on fatal errors

    Args:
        abort_code (int)
    """
    ...


class BabSolution:
    @property
    def X(self) -> npt.NDArray[np.int32]: ...


class BabNode:
    @property
    def xfixed(self) -> npt.NDArray[np.int32]: ...
    @property
    def sol(self) -> BabSolution: ...
    @property
    def level(self) -> int: ...
    @property
    def upper_bound(self) -> float: ...
    # fracsol is mutable — SDPbound writes back into it
    @property
    def fracsol(self) -> npt.NDArray[np.float64]: ...
    @fracsol.setter
    def fracsol(self, value: npt.NDArray[np.float64]) -> None: ...


class Problem:
    @property
    def L(self) -> npt.NDArray[np.float64]: ...  # shape: (n, n)
    @property
    def n(self) -> int: ...
    @property
    def NIneq(self) -> int: ...
    @property
    def NPentIneq(self) -> int: ...
    @property
    def NHeptaIneq(self) -> int: ...
    @property
    def bundle(self) -> int: ...
