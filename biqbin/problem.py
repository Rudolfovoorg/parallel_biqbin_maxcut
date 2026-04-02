import numpy as np
import numpy.typing as npt

from biqbin.utils import check_matrix_validity_wrap, divide_matrix_by_gcd, PrettyPrint


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
    """Inherits from ProblemMaxCut, takes a qubo np.ndarray and constructs Biqbin input (maxcut adjacency matrix)
    """

    def __init__(self, Q: np.ndarray, offset: float, problem_name: str, is_minimization: bool = True, optimize_input: bool = False):
        """Initialize the ProblemQubo

        Args:
            Q (np.ndarray): Qubo in a dense matrix.
            problem_name (str): Name (filename) of the problem instance.
            is_minimization (bool): If the objective is to minimize (True) or maximize (False). Defaults to True.
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
        return (f'Class: {type(self).__name__}\n'
                f'Problem name = {self.problem_name}\n'
                f'Offset = {self.offset}\n'
                f'Minimizing = {self.is_minimization}\n'
                f'Qubo {self.Q.shape} =\n{self.Q}\n')
