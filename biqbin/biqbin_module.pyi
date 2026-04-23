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
