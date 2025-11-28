# biqbin.pyi
import numpy as np
from typing import Callable

"""biqbin.so
"""
def set_heuristic(heuristic_function: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]):
    """Sets the heuristic function in biqbin."""
    ...

def set_read_data(read_data_function: Callable[[], np.ndarray]):
    """Sets the read_data function in biqbin."""
    ...

def run(solver_name: str, problem_instance_path: str, params_path: str, time_limit: int) -> dict:
    """Runs the solver, returns a solution dictionary on mpi rank 0"""
    ...

def default_heuristic(L0: np.ndarray, L: np.ndarray, xfixed: np.ndarray, sol_X: np.ndarray, x: np.ndarray) -> float:
    """Runs the default GW heuristic."""
    ...

def default_read_data(problem_instance_path: str) -> np.ndarray:
    """Runs the default read_data returns the adj matrix as numpy ndarray."""
    ...

def get_rank() -> int:
    """Returns the MPI rank."""
    ...
