# biqbin.pyi
import numpy as np
import numpy.typing as npt
from typing import Callable, Tuple

"""biqbin.so
"""


def run(solver_name: str, problem_instance_path: str, maxcut_adj_matrix: npt.NDArray[np.float64] | None, params_path: str, time_limit: int) -> dict:
    """Runs the solver, returns a solution dictionary on mpi rank 0"""
    ...


def set_initial_solution(x: np.ndarray):
    """Set the initial solution

    Args:
        x (np.ndarray): binary vector as numpy array
    """


def set_heuristic(heuristic_function: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]):
    """Sets the heuristic function in biqbin."""
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
    """Runs the default GW heuristic."""
    ...


def set_node_evaluation(node_eval_function: Callable[[BabNode, Problem, Problem, int], float]):
    """Set the function that will evaluate both bounds.
    """
    ...


def sdp_bound(node: BabNode, main_problem: Problem, subproblem: Problem, rank: int) -> float:
    """Default SDPBound in src/bounding.c. Computes both bound...

    Args:
        L0 (np.ndarray): _description_
        L (np.ndarray): _description_
        xfixed (np.ndarray): _description_
        sol_X (np.ndarray): _description_
        fracsol (np.ndarray): _description_
        rank (int): _description_

    Returns:
        float: upper bound (maximization), lower bound (minimization)
    """
    ...


def get_rank() -> int:
    """Returns the MPI rank."""
    ...


def init_mpi() -> Tuple[int, int]:
    """initialize MPI

    Returns:
        (int, int): MPI (size, rank) tuple 
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
