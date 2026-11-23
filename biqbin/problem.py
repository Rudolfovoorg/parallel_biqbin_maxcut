"""Problem definitions and their Max-Cut reformulations."""

import numpy as np
import numpy.typing as npt

from biqbin.biqbin_module import interior_point_method_maxcut
from biqbin.configuration import Option
from biqbin.logging import logger
from biqbin.utils import PrettyPrint, check_matrix_validity_setter_wrap, divide_matrix_by_gcd


class ProblemMaxCut(PrettyPrint):
    """Weighted Max-Cut problem with an integer-valued symmetric adjacency matrix.

    Args:
        maxcut_adjacency_matrix: Square matrix of edge weights with zero diagonal.
        problem_name: Name used in logging and result metadata.
        optimize_mc_adj_matrix: Divide weights by their greatest common divisor.
            This may modify the supplied matrix; pass a copy to preserve it.
    """

    @classmethod
    def config_options(cls) -> dict[str, Option]:
        """Configuration owned by problem preprocessing, shared by all problem types."""
        return {
            "optimize": Option(
                bool,
                default=False,
                help="Divide transformed Max-Cut weights by their greatest common divisor",
                aliases=("-O", "--optimize"),
            ),
        }

    def __init__(
        self,
        maxcut_adjacency_matrix: npt.NDArray[np.floating | np.integer],
        problem_name: str,
        optimize_mc_adj_matrix: bool = False,
    ) -> None:

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

    def _summary_fields(self) -> list[tuple[str, object]]:
        return [
            ("Problem name", self.problem_name),
            ("Optimized input", self.optimize_mc_adj_matrix),
            ("GCD", self.gcd),
        ]

    def _format_maxcut_problem(self, title: str = "Max-Cut") -> str:
        return "\n".join(
            [
                f"--- {title} ---",
                self._format_fields([("Vertices", self.maxcut_adjacency_matrix.shape[0])]),
                self._format_array("Adjacency matrix", self.maxcut_adjacency_matrix),
            ]
        )

    def _summary_sections(self, verbose: bool) -> list[str]:
        return [self._format_maxcut_problem()]


class ProblemQubo(ProblemMaxCut):
    """Inherits from ProblemMaxCut, takes a qubo np.ndarray and constructs the biqbin input (maxcut adjacency matrix)"""

    def __init__(
        self,
        Q: np.ndarray,
        offset: float,
        problem_name: str,
        is_minimization: bool,
        optimize_input: bool = False,
    ):
        """Initialize the ProblemQubo

        Args:
            Q (np.ndarray): Dense matrix defining the objective ``x.T @ Q @ x + offset``.
            offset (float): Constant added to the objective value.
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
        """Convert a QUBO matrix to a Max-Cut instance.

        QUBO coefficients must be integer-valued. Since the conversion
        symmetrizes Q as (Q + Q.T) / 2, triangular QUBO inputs must have
        even off-diagonal coefficients so that the resulting Max-Cut
        weights remain integer-valued.

        Raises:
            ValueError: If the resulting Max-Cut matrix is not integer-valued.

        Args:
            qubo (np.ndarray): qubo as 2D numpy array
        Returns:
            np.ndarray: adjacency matrix for max cut problem
        """

        q_sym = 1 / 2 * (qubo.T + qubo)

        Qe_plus_c = -np.array([(np.sum(q_sym, 1))])
        np.fill_diagonal(q_sym, 0)

        return np.block([[q_sym, Qe_plus_c.T], [Qe_plus_c, np.zeros((1, 1))]])

    def _summary_sections(self, verbose: bool) -> list[str]:
        sections = [
            "\n".join(
                [
                    "--- QUBO ---",
                    self._format_fields(
                        [
                            ("Variables", self.Q.shape[0]),
                            ("Minimization", self.is_minimization),
                            ("Offset", self.offset),
                        ]
                    ),
                    self._format_array("Q", self.Q),
                ]
            )
        ]
        if verbose:
            sections.append(self._format_maxcut_problem("Underlying Max-Cut"))
        return sections


class ProblemBQP(ProblemMaxCut):
    """
    Linearly constrained binary quadratic optimization problem.

    ```
    min         x.T @ F @ x + c.T @ x
    subject to  A @ x = b
                x_i in {0, 1}
    ```
    """

    def __init__(
        self,
        F: np.ndarray,
        c: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        problem_name: str,
        offset: int = 0,
        optimize_input: bool = False,
    ):
        """
        All values in the inputs `F`, `c`, `A` and `b` need to have integer values.

        ```
        min         x.T @ F @ x + c.T @ x + offset
        subject to  A @ x = b
                    x_i in {0, 1}
        ```

        Args:
            F (np.ndarray): Quadratic objective matrix. It is expected to be a symmetric matrix.
            c (np.ndarray): Linear objective vector.
            A (np.ndarray): Equality-constraint matrix.
            b (np.ndarray): Equality-constraint vector.
            problem_name (str): Human-readable name of the problem.
            offset (int): Constant added to the original objective.
            optimize_input (bool): Optimizes the input for the solver.
        """
        fatal_error = False

        if F.ndim != 2 or F.shape[0] != F.shape[1]:
            logger.fatal(f"Matrix F must be square, got {F.shape}")
            fatal_error = True
            n = None
        else:
            n = F.shape[0]
            if not np.array_equal(F, F.T):
                logger.fatal("Matrix F must be symmetric")
                fatal_error = True
        if not np.all(np.isfinite(F)) or not np.array_equal(F, np.round(F)):
            logger.fatal("All values in matrix F must be integers!")
            fatal_error = True

        if n is not None and (c.ndim != 1 or c.shape != (n,)):
            logger.fatal(f"Vector c expected shape is ({n},), got {c.shape}")
            fatal_error = True
        if not np.all(np.isfinite(c)) or not np.array_equal(c, np.round(c)):
            logger.fatal("All values in vector c must be integers!")
            fatal_error = True

        if A.ndim != 2:
            logger.fatal(f"Matrix A must be 2-dimensional, got {A.shape}")
            fatal_error = True
        elif n is not None and A.shape[1] != n:
            logger.fatal(f"Matrix A expected {n} columns, got {A.shape}")
            fatal_error = True

        if not np.all(np.isfinite(A)) or not np.array_equal(A, np.round(A)):
            logger.fatal("All values in matrix A must be integers!")
            fatal_error = True

        if A.ndim == 2 and (b.ndim != 1 or b.shape != (A.shape[0],)):
            logger.fatal(f"Vector b expected shape is ({A.shape[0]},), got {b.shape}")
            fatal_error = True
        if not np.all(np.isfinite(b)) or not np.array_equal(b, np.round(b)):
            logger.fatal("All values in vector b must be integers!")
            fatal_error = True

        if fatal_error:
            raise ValueError("Invalid BQP problem.")

        self.F: np.ndarray = F.astype(np.int64)
        self.c: np.ndarray = c.astype(np.int64)
        self.A: np.ndarray = A.astype(np.int64)
        self.b: np.ndarray = b.astype(np.int64)

        self.offset = offset
        maxcut_adj, const_val, penalty, rho = self.process_bqp_input()

        self.penalty: float = penalty
        self.rho: float = rho
        self.const_value: float = const_val

        super().__init__(maxcut_adj, problem_name, optimize_input)

    def _summary_sections(self, verbose: bool) -> list[str]:
        sections = [
            "\n".join(
                [
                    "--- BQP ---",
                    self._format_fields(
                        [
                            ("Variables", self.F.shape[0]),
                            ("Equality constraints", self.A.shape[0]),
                            ("Offset", self.offset),
                        ]
                    ),
                    self._format_array("F", self.F),
                    self._format_array("c", self.c),
                    self._format_array("A", self.A),
                    self._format_array("b", self.b),
                ]
            )
        ]
        if verbose:
            sections.append(self._format_maxcut_problem("Underlying Max-Cut"))
            sections.append(
                "--- BQP transformation ---\n"
                + self._format_fields(
                    [
                        ("Rho", self.rho),
                        ("Const value", self.const_value),
                        ("Penalty", self.penalty),
                    ]
                )
            )
        return sections

    def process_bqp_input(self) -> tuple[np.ndarray, int, int, float]:
        """
        Transform a linearly constrained BQP into a Max-Cut adjacency matrix.

        Assumes:
            ipm_mc_pk(C) -> maximum value of <C, X>
            subject to diag(X) = 1, X >= 0

        Returns:
            adj (np.ndarray): Max-Cut adjacency matrix
            const_val (int):  Recover original value with: original_value = const_val - max_cut_value
            penalty (int):    Exact penalty parameter.
            rho (float):      Max absolute SDP solution value
        """
        n = self.F.shape[0]
        e = np.ones(n)

        # Change problem variables from {0, 1} to {-1, 1}
        sdp_constant = 0.25 * e @ self.F @ e + 0.5 * self.c @ e

        # Computing the penalty parameter
        A_scaled = 0.5 * self.A
        b_shifted = self.b - A_scaled @ e
        c_shifted = 0.5 * (self.F @ e + self.c)
        F_scaled = 0.25 * self.F

        # SDP matrix
        C = np.block(
            [
                [F_scaled, 0.5 * c_shifted[:, None]],
                [0.5 * c_shifted[None, :], np.array([[sdp_constant]])],
            ]
        )

        # SDP maximum and minimum.
        r_max, _ = interior_point_method_maxcut(C)
        r_min, _ = interior_point_method_maxcut(-C)

        rho = max(abs(r_min), abs(r_max))
        penalty = int(np.ceil(2 * rho + 1))

        # Penalized quadratic matrix.
        top_left = F_scaled + penalty * A_scaled.T @ A_scaled
        top_right = 0.5 * c_shifted - penalty * A_scaled.T @ b_shifted
        bottom_right = sdp_constant + penalty * (b_shifted @ b_shifted)

        B = np.block(
            [
                [top_left, top_right[:, None]],
                [top_right[None, :], np.array([[bottom_right]])],
            ]
        )

        # original_value = const_val - max_cut_value
        #
        # Derivation: maxcut_adjacency = 4*B with zero diagonal, and for such a
        # zero-diagonal weight matrix M, cut_value = 1/4 * (M.sum() - y.T @ M @ y)
        # where y = (x, 1).
        #
        # Substituting M = 4*B (diagonal cancels since y_i**2 == 1) gives
        # y.T @ B @ y == B.sum() - cut_value, i.e. B.sum() is exactly the
        # constant to recover the original objective.

        const_val = B.sum()
        if not np.isclose(const_val, round(const_val)):
            raise ValueError(f"Constant must be an integer, got {const_val}")
        const_val = round(const_val)

        # Standard weighted Max-Cut adjacency matrix.
        maxcut_adjacency = 4 * B
        np.fill_diagonal(maxcut_adjacency, 0)

        return maxcut_adjacency, const_val, penalty, rho
