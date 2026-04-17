# biqbin.pyi
import numpy as np
import numpy.typing as npt
from typing import Callable, Tuple

"""biqbin.so
"""


def run(solver_name: str, problem_instance_path: str, maxcut_adj_matrix: npt.NDArray[np.float64] | None, params: _Parameters) -> dict:
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


def goemans_williamson_heuristic(L0: np.ndarray, L: np.ndarray, xfixed: np.ndarray, sol_X: np.ndarray, x: np.ndarray) -> float:
    """Runs the default GW heuristic."""
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


class _Parameters:
    """ BiqbinParameters c-struct, located in src/parameters.h
    """
        # Bundle iterations
    init_bundle_iter: int
    max_bundle_iter: int

    # Cutting plane iterations
    triag_iter: int
    pent_iter: int
    hept_iter: int

    # Outer loop
    max_outer_iter: int
    extra_iter: int

    # Triangle inequalities
    violated_TriIneq: float
    TriIneq: int
    adjust_TriIneq: int

    # Pentagon inequalities
    PentIneq: int
    Pent_Trials: int
    include_Pent: int

    # Heptagon inequalities
    HeptaIneq: int
    Hepta_Trials: int
    include_Hepta: int

    # Solver behaviour
    root: int
    use_diff: int
    time_limit: int
    branchingStrategy: int