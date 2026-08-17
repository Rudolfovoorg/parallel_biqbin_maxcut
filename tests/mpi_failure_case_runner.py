from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def make_problem():
    from biqbin import ProblemMaxCut

    A = np.array(
        [
            [0, 1, 1, 0, 1],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 1, 1],
            [0, 1, 1, 0, 1],
            [1, 0, 1, 1, 0],
        ],
        dtype=np.float64,
    )

    return ProblemMaxCut(A, problem_name="mpi_failure_case")


def write_marker(log_dir: Path, case: str, rank: int, marker: str) -> None:
    """
    Write a small marker file before intentionally failing.

    The fsync is deliberate: MPI_Abort or native crashes can kill the process
    immediately after this point.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    path = log_dir / f"{case}.rank{rank}.{marker}.marker"

    with path.open("w", encoding="utf-8") as f:
        f.write(marker + "\n")
        f.flush()
        os.fsync(f.fileno())

    # Also print it for easier manual debugging.
    print(marker, flush=True)


def solver_class_for(case: str, log_dir: Path):
    from biqbin import MaxCutSolver
    from biqbin.biqbin_module import abort_mpi

    class BaseSolver(MaxCutSolver):
        def mark(self, marker: str) -> None:
            write_marker(log_dir, case, self.rank, marker)

        def good_sdp_bound(self, node, P0, P) -> float:
            self.set_sdp_primal_solution(np.eye(P.n, dtype=np.float64))
            return 1.0e9

    # ------------------------------------------------------------------
    # Explicit abort and raw Python callback exception tests.
    # ------------------------------------------------------------------

    if case == "callback_abort_rank0":
        class Solver(BaseSolver):  # type: ignore
            def sdp_bound(self, node, P0, P, *args, **kwargs):
                if self.rank == 0:
                    self.mark("BIQBIN_TEST_CALLBACK_ABORT_RANK0")
                    abort_mpi(10)
                    raise SystemExit(10)

                return self.good_sdp_bound(node, P0, P)

        return Solver

    if case == "callback_exception_rank0":
        class Solver(BaseSolver):  # type: ignore
            def sdp_bound(self, node, P0, P, *args, **kwargs):
                if self.rank == 0:
                    self.mark("BIQBIN_TEST_CALLBACK_EXCEPTION_RANK0")
                    raise RuntimeError("BIQBIN_TEST_CALLBACK_EXCEPTION_RANK0")

                return self.good_sdp_bound(node, P0, P)

        return Solver

    # ------------------------------------------------------------------
    # Exception handling at each Python callback boundary.
    # ------------------------------------------------------------------

    if case == "initial_sdp_exception_rank0":
        class Solver(BaseSolver):  # type: ignore
            def initial_sdp_bound(self, node, P0, P, *args, **kwargs):
                if self.rank == 0:
                    self.mark("BIQBIN_TEST_INITIAL_SDP_EXCEPTION_RANK0")
                    raise RuntimeError(
                        "BIQBIN_TEST_INITIAL_SDP_EXCEPTION_RANK0")

                return self.good_sdp_bound(node, P0, P)

        return Solver

    if case == "sdp_exception_rank0":
        class Solver(BaseSolver):  # type: ignore
            def sdp_bound(self, node, P0, P, *args, **kwargs):
                if self.rank == 0:
                    self.mark("BIQBIN_TEST_SDP_EXCEPTION_RANK0")
                    raise RuntimeError("BIQBIN_TEST_SDP_EXCEPTION_RANK0")

                return self.good_sdp_bound(node, P0, P)

        return Solver

    if case == "initial_heuristic_exception_rank0":
        class Solver(BaseSolver):  # type: ignore
            def initial_heuristic(self, L, *args, **kwargs):
                if self.rank == 0:
                    self.mark("BIQBIN_TEST_INITIAL_HEURISTIC_EXCEPTION_RANK0")
                    raise RuntimeError(
                        "BIQBIN_TEST_INITIAL_HEURISTIC_EXCEPTION_RANK0")

                P = kwargs["P"]
                return np.zeros(P.n - self._MC_OFFSET, dtype=np.int32)

        return Solver

    if case == "heuristic_exception_rank0":
        class Solver(BaseSolver):  # type: ignore
            def heuristic(self, L, *args, **kwargs):
                if self.rank == 0:
                    self.mark("BIQBIN_TEST_HEURISTIC_EXCEPTION_RANK0")
                    raise RuntimeError("BIQBIN_TEST_HEURISTIC_EXCEPTION_RANK0")

                P = kwargs["P"]
                return np.zeros(P.n - self._MC_OFFSET, dtype=np.int32)

        return Solver

    # ------------------------------------------------------------------
    # Initial SDP return-value validation.
    # ------------------------------------------------------------------

    if case == "initial_sdp_returns_invalid_bound":
        class Solver(BaseSolver):  # type: ignore
            def initial_sdp_bound(self, node, P0, P, *args, **kwargs):
                self.mark("BIQBIN_TEST_INITIAL_SDP_RETURNS_INVALID_BOUND")
                self.set_sdp_primal_solution(np.eye(P.n, dtype=np.float64))
                return -10.0

        return Solver

    if case == "initial_sdp_returns_nan":
        class Solver(BaseSolver):  # type: ignore
            def initial_sdp_bound(self, node, P0, P, *args, **kwargs):
                self.mark("BIQBIN_TEST_INITIAL_SDP_RETURNS_NAN")
                self.set_sdp_primal_solution(np.eye(P.n, dtype=np.float64))
                return float("nan")

        return Solver

    if case == "initial_sdp_returns_inf":
        class Solver(BaseSolver):  # type: ignore
            def initial_sdp_bound(self, node, P0, P, *args, **kwargs):
                self.mark("BIQBIN_TEST_INITIAL_SDP_RETURNS_INF")
                self.set_sdp_primal_solution(np.eye(P.n, dtype=np.float64))
                return float("inf")

        return Solver

    if case == "initial_sdp_missing_primal":
        class Solver(BaseSolver):  # type: ignore
            def initial_sdp_bound(self, node, P0, P, *args, **kwargs):
                self.mark("BIQBIN_TEST_INITIAL_SDP_MISSING_PRIMAL")
                return 1.0e9

        return Solver

    # ------------------------------------------------------------------
    # Non-initial SDP return-value validation.
    # ------------------------------------------------------------------

    if case == "sdp_returns_nan":
        class Solver(BaseSolver):  # type: ignore
            def sdp_bound(self, node, P0, P, *args, **kwargs):
                self.mark("BIQBIN_TEST_SDP_RETURNS_NAN")
                self.set_sdp_primal_solution(np.eye(P.n, dtype=np.float64))
                return float("nan")

        return Solver

    if case == "sdp_returns_inf":
        class Solver(BaseSolver):  # type: ignore
            def sdp_bound(self, node, P0, P, *args, **kwargs):
                self.mark("BIQBIN_TEST_SDP_RETURNS_INF")
                self.set_sdp_primal_solution(np.eye(P.n, dtype=np.float64))
                return float("inf")

        return Solver

    if case == "sdp_returns_none":
        class Solver(BaseSolver):  # type: ignore
            def sdp_bound(self, node, P0, P, *args, **kwargs):  # type:ignore
                self.mark("BIQBIN_TEST_SDP_RETURNS_NONE")
                self.set_sdp_primal_solution(np.eye(P.n, dtype=np.float64))
                return None

        return Solver

    if case == "sdp_returns_string":
        class Solver(BaseSolver):  # type: ignore
            def sdp_bound(self, node, P0, P, *args, **kwargs):  # type:ignore
                self.mark("BIQBIN_TEST_SDP_RETURNS_STRING")
                self.set_sdp_primal_solution(np.eye(P.n, dtype=np.float64))
                return "not-a-bound"

        return Solver

    if case == "sdp_returns_array":
        class Solver(BaseSolver):  # type: ignore
            def sdp_bound(self, node, P0, P, *args, **kwargs):  # type:ignore
                self.mark("BIQBIN_TEST_SDP_RETURNS_ARRAY")
                self.set_sdp_primal_solution(np.eye(P.n, dtype=np.float64))
                return np.array([1.0], dtype=np.float64)

        return Solver

    if case == "sdp_rank1_rank2_returns_invalid_bound":
        class Solver(BaseSolver):  # type: ignore
            def sdp_bound(self, node, P0, P, *args, **kwargs):
                if self.rank in (1, 2):
                    self.mark(
                        "BIQBIN_TEST_RANK1_RANK2_SDP_RETURNS_INVALID_BOUND")
                    self.set_sdp_primal_solution(np.eye(P.n, dtype=np.float64))
                    return -10.0

                return self.good_sdp_bound(node, P0, P)

        return Solver

    # ------------------------------------------------------------------
    # SDP primal-solution validation.
    # ------------------------------------------------------------------

    if case == "sdp_wrong_primal_shape":
        class Solver(BaseSolver):  # type: ignore
            def sdp_bound(self, node, P0, P, *args, **kwargs):
                self.mark("BIQBIN_TEST_SDP_WRONG_PRIMAL_SHAPE")
                self.set_sdp_primal_solution(np.eye(P.n + 1, dtype=np.float64))
                return 1.0e9

        return Solver

    if case == "sdp_missing_primal":
        class Solver(BaseSolver):  # type: ignore
            def sdp_bound(self, node, P0, P, *args, **kwargs):
                self.mark("BIQBIN_TEST_SDP_MISSING_PRIMAL")
                return 1.0e9

        return Solver

    if case == "sdp_primal_contains_nan":
        class Solver(BaseSolver):  # type: ignore
            def sdp_bound(self, node, P0, P, *args, **kwargs):
                self.mark("BIQBIN_TEST_SDP_PRIMAL_CONTAINS_NAN")

                X = np.eye(P.n, dtype=np.float64)
                X[0, 0] = np.nan

                self.set_sdp_primal_solution(X)
                return 1.0e9

        return Solver

    if case == "sdp_primal_contains_inf":
        class Solver(BaseSolver):  # type: ignore
            def sdp_bound(self, node, P0, P, *args, **kwargs):
                self.mark("BIQBIN_TEST_SDP_PRIMAL_CONTAINS_INF")

                X = np.eye(P.n, dtype=np.float64)
                X[0, 0] = np.inf

                self.set_sdp_primal_solution(X)
                return 1.0e9

        return Solver

    if case == "sdp_primal_out_of_range":
        class Solver(BaseSolver):  # type: ignore
            def sdp_bound(self, node, P0, P, *args, **kwargs):
                self.mark("BIQBIN_TEST_SDP_PRIMAL_OUT_OF_RANGE")

                X = np.eye(P.n, dtype=np.float64)
                X[0, 1] = 2.0
                X[1, 0] = 2.0

                self.set_sdp_primal_solution(X)
                return 1.0e9

        return Solver

    if case == "sdp_primal_not_2d":
        class Solver(BaseSolver):  # type: ignore
            def sdp_bound(self, node, P0, P, *args, **kwargs):
                self.mark("BIQBIN_TEST_SDP_PRIMAL_NOT_2D")

                X = np.zeros(P.n, dtype=np.float64)

                self.set_sdp_primal_solution(X)
                return 1.0e9

        return Solver

    if case == "sdp_primal_non_square":
        class Solver(BaseSolver):  # type: ignore
            def sdp_bound(self, node, P0, P, *args, **kwargs):
                self.mark("BIQBIN_TEST_SDP_PRIMAL_NON_SQUARE")

                X = np.zeros((P.n, P.n + 1), dtype=np.float64)

                self.set_sdp_primal_solution(X)
                return 1.0e9

        return Solver

    # ------------------------------------------------------------------
    # Heuristic solution validation.
    # ------------------------------------------------------------------

    if case == "heuristic_wrong_length":
        class Solver(BaseSolver):  # type: ignore
            def heuristic(self, L, *args, **kwargs):
                self.mark("BIQBIN_TEST_HEURISTIC_WRONG_LENGTH")

                P = kwargs["P"]
                return np.zeros(P.n - self._MC_OFFSET - 1, dtype=np.int32)

        return Solver

    if case == "heuristic_non_binary01_vector":
        class Solver(BaseSolver):  # type: ignore
            def heuristic(self, L, *args, **kwargs):
                self.mark("BIQBIN_TEST_HEURISTIC_NON_BINARY01_VECTOR")

                P = kwargs["P"]

                # Integer dtype, invalid value.
                return np.full(P.n - self._MC_OFFSET, 2, dtype=np.int32)

        return Solver

    # ------------------------------------------------------------------
    # Initial heuristic validation.
    # ------------------------------------------------------------------

    if case == "initial_heuristic_wrong_length":
        class Solver(BaseSolver):  # type: ignore
            def initial_heuristic(self, L, *args, **kwargs):
                self.mark("BIQBIN_TEST_INITIAL_HEURISTIC_WRONG_LENGTH")

                P = kwargs["P"]
                return np.zeros(P.n - self._MC_OFFSET - 1, dtype=np.int32)

        return Solver

    if case == "initial_heuristic_non_binary01_vector":
        class Solver(BaseSolver):  # type: ignore
            def initial_heuristic(self, L, *args, **kwargs):
                self.mark("BIQBIN_TEST_INITIAL_HEURISTIC_NON_BINARY01_VECTOR")

                P = kwargs["P"]

                # Integer dtype, invalid value.
                return np.full(P.n - self._MC_OFFSET, 2, dtype=np.int32)

        return Solver

    # ------------------------------------------------------------------
    # Optional read-only native view tests.
    #
    # Keep these only if Problem/BabNode arrays are supposed to be read-only.
    # ------------------------------------------------------------------

    if case == "callback_mutates_problem_matrix":
        class Solver(BaseSolver):  # type: ignore
            def sdp_bound(self, node, P0, P, *args, **kwargs):
                self.mark("BIQBIN_TEST_MUTATE_PROBLEM_MATRIX")

                P.L[0, 0] = 123.0

                return self.good_sdp_bound(node, P0, P)

        return Solver

    if case == "callback_mutates_node_solution":
        class Solver(BaseSolver):  # type: ignore
            def sdp_bound(self, node, P0, P, *args, **kwargs):
                self.mark("BIQBIN_TEST_MUTATE_NODE_SOLUTION")

                node.sol.x[0] = 1

                return self.good_sdp_bound(node, P0, P)

        return Solver

    raise ValueError(f"unknown failure case: {case}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--log-dir", required=True)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)

    import biqbin

    biqbin.init()

    problem = make_problem()
    Solver = solver_class_for(args.case, log_dir)

    solver = Solver(problem=problem)
    solver.compute()

    print(f"UNEXPECTED_SUCCESS:{args.case}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
