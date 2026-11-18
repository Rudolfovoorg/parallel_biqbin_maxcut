"""Solver results in Max-Cut, QUBO, and BQP variable spaces."""

from typing import Any

import numpy as np

from biqbin.problem import ProblemBQP, ProblemMaxCut, ProblemQubo
from biqbin.utils import PrettyPrint


class SolutionMaxCut(PrettyPrint):
    """Max-Cut result returned by the solver on MPI rank 0.

    ``solution`` contains ``computed_val``, the binary partition vector ``x``,
    and ``solution`` (the 1-based indices where ``x`` equals one).
    ``meta_data`` contains timings, search statistics, and root-node diagnostics.
    A time-limited run can return an incumbent; check ``time_limit_reached``
    in ``meta_data`` before treating the result as proven optimal.
    """

    def __init__(self, biqbin_result: dict, problem: ProblemMaxCut) -> None:
        if problem.optimize_mc_adj_matrix and problem.gcd != 1:
            biqbin_result["maxcut"]["computed_val"] *= problem.gcd

        self.__problem: ProblemMaxCut = problem
        self.__solution: dict = biqbin_result["maxcut"]
        self.meta_data: dict = biqbin_result["meta_data"]

    @property
    def solution(self) -> dict:
        return self.__solution

    @property
    def problem(self) -> ProblemMaxCut:
        return self.__problem

    def _summary_fields(self) -> list[tuple[str, object]]:
        """Common run statistics are visible at every verbosity."""
        metadata = self.meta_data
        return [
            ("Problem name", metadata["instance"]),
            ("Compute time", f"{metadata['time']:.2f} seconds"),
            ("Time limit reached", metadata["time_limit_reached"]),
            ("B&B nodes evaluated", metadata["eval_bab_nodes"]),
            ("Heuristic calls", metadata["heuristic_call_count"]),
            ("Worker processes used", metadata["num_workers_used"]),
            ("Optimized input", metadata["parameters"]["optimized"]),
            ("GCD", metadata["parameters"]["gcd"]),
        ]

    def _format_maxcut_section(self, title: str = "Max-Cut") -> str:
        return f"--- {title} ---\n" + self._format_fields(
            [
                ("Computed value", self.__solution["computed_val"]),
                ("Solution", self.__solution["solution"]),
                ("x", self.__solution["x"]),
            ]
        )

    def _summary_sections(self, verbose: bool) -> list[str]:
        return [self._format_maxcut_section()]

    def print_computed_solution(self, verbose: bool = False) -> None:
        """Print the shared summary, optionally including transformation details."""
        print(self.format_summary(verbose))


class SolutionQubo(SolutionMaxCut):
    """QUBO result in the original problem's objective and variable space.

    ``solution`` contains ``computed_val`` (including the offset), binary ``x``,
    1-based selected indices in ``solution``, ``cardinality``, ``offset``, and
    ``minimization``. ``solution_maxcut`` retains the transformed Max-Cut result.
    """

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

        computed_val = float(problem.Q.dot(qubo_x).dot(qubo_x)) + self.problem.offset
        return {
            "computed_val": computed_val,
            "offset": self.problem.offset,
            "solution": qubo_solution,
            "x": qubo_x,
            "cardinality": float(sum(qubo_x)),
            "minimization": problem.is_minimization,
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

    def _summary_sections(self, verbose: bool) -> list[str]:
        sections = [
            "--- QUBO ---\n"
            + self._format_fields(
                [
                    ("Computed value", self.__solution["computed_val"]),
                    ("Minimization", self.__problem.is_minimization),
                    ("Solution", self.__solution["solution"]),
                    ("x", self.__solution["x"]),
                ]
            )
        ]
        if verbose:
            sections.append(self._format_maxcut_section("Underlying Max-Cut"))
        return sections


class SolutionBQP(SolutionMaxCut):
    """BQP result reconstructed from the penalized Max-Cut solution.

    ``solution["feasible_solution"]`` indicates whether a feasible assignment
    was recovered. When true, ``x`` is binary, satisfies ``A @ x == b``, and
    ``computed_val`` includes the problem offset. Otherwise both are ``None``.
    ``maxcut_solution`` retains the underlying Max-Cut result.
    """

    def __init__(self, biqbin_result: dict, problem: ProblemBQP) -> None:
        super().__init__(biqbin_result, problem)
        self.maxcut_solution = super().solution
        self.__solution = self.get_bqp_solution(biqbin_result, problem)

    @property
    def solution(self):
        return self.__solution

    def get_bqp_solution(self, biqbin_result: dict, problem: ProblemBQP) -> dict[str, Any]:
        """Construct BQP solution from Biqbins Max-Cut solution

        Args:
            biqbin_result (dict): raw result retrieved from native Biqbin solver
            problem (ProblemBQP): BQP problem being solved

        Raises:
            ValueError: If the penalty test says the solution should be feasible but
                        neither Max-Cut orientation maps back to a valid BQP solution

        Returns:
            dict[str, Any]: Result dict with added BQP solution under the 'bqp' key
        """

        bqp_result: dict[str, Any] = {
            "rho": problem.rho,
            "const_value": problem.const_value,
            "penalty": problem.penalty,
            "feasible_solution": False,
            "computed_val": None,
            "x": None,
        }

        biqbin_result["bqp"] = bqp_result

        mc_obj_value = biqbin_result["maxcut"]["computed_val"]

        # The Max-Cut solution has one additional auxiliary variable.
        # Remove it to recover the BQP-sized vector.
        mc_x = np.asarray(biqbin_result["maxcut"]["x"])[:-1]

        # Recover the objective value of the penalized BQP from Max-Cut.
        bqp_obj_value = problem.const_value - mc_obj_value

        # By construction of the exact penalty, a value above rho means that
        # the returned solution corresponds to an infeasible BQP assignment.
        if bqp_obj_value > problem.rho:
            return bqp_result

        # A Max-Cut partition is invariant under a global bit flip:
        # x and 1 - x represent the same cut.
        #
        # The original BQP is not invariant under this flip, so try both
        # orientations and select the one that satisfies the original
        # constraints and reproduces the recovered BQP objective.
        for candidate in (mc_x, 1 - mc_x):
            constraints_ok = np.array_equal(problem.A @ candidate, problem.b)

            if not constraints_ok:
                continue

            objective = candidate @ problem.F @ candidate + problem.c @ candidate

            if np.isclose(objective, bqp_obj_value):
                bqp_result["x"] = candidate.tolist()
                bqp_result["computed_val"] = bqp_obj_value + problem.offset
                bqp_result["feasible_solution"] = True
                bqp_result["offset"] = problem.offset
                return bqp_result

        # If the penalty test says the solution should be feasible but neither
        # Max-Cut orientation maps back to a valid BQP solution, something is
        # inconsistent in the transformation/reconstruction.
        raise ValueError(
            "Max-Cut solution could not be mapped back to a feasible BQP "
            "solution with the expected objective value."
        )

    def _summary_sections(self, verbose: bool) -> list[str]:
        sections = [
            "--- BQP ---\n"
            + self._format_fields(
                [
                    ("Computed value", self.__solution["computed_val"]),
                    ("Feasible solution", self.__solution["feasible_solution"]),
                    ("x", self.__solution["x"]),
                ]
            )
        ]
        if verbose:
            sections.append(
                "--- BQP transformation ---\n"
                + self._format_fields(
                    [
                        ("Rho", self.__solution["rho"]),
                        ("Const value", self.__solution["const_value"]),
                        ("Penalty", self.__solution["penalty"]),
                    ]
                )
            )
            sections.append(self._format_maxcut_section("Underlying Max-Cut"))
        return sections
