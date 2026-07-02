# tests/mpi_failure_case_runner.py

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

    class BaseSolver(MaxCutSolver):
        def mark(self, marker: str) -> None:
            write_marker(log_dir, case, self.rank, marker)

        def good_sdp_bound(self, node, P0, P) -> float:
            self.set_sdp_primal_solution(np.eye(P.n, dtype=np.float64))
            return 1.0e9

        def heuristic(self, L, *args, **kwargs):
            P = kwargs["P"]
            return np.zeros(P.n - self._MC_OFFSET, dtype=np.int32)

    if case == "callback_exception_rank0":
        class Solver(BaseSolver):  # type: ignore
            def sdp_bound(self, node, P0, P, *args, **kwargs):
                if self.rank == 0:
                    self.mark("BIQBIN_TEST_CALLBACK_EXCEPTION_RANK0")
                    raise RuntimeError("BIQBIN_TEST_CALLBACK_EXCEPTION_RANK0")

                return self.good_sdp_bound(node, P0, P)

        return Solver

    if case == "initial_sdp_returns_invalid_bound":
        class Solver(BaseSolver):  # type: ignore
            def initial_sdp_bound(self, node, P0, P, *args, **kwargs):
                self.mark("BIQBIN_TEST_INITIAL_SDP_RETURNS_INVALID_BOUND")
                self.set_sdp_primal_solution(np.eye(P.n, dtype=np.float64))
                return float("nan")

        return Solver

    if case == "sdp_rank1_rank2_returns_invalid_bound":
        class Solver(BaseSolver):  # type: ignore
            def initial_sdp_bound(self, node, P0, P, *args, **kwargs) -> float:
                return self.good_sdp_bound(node, P0, P, *args, **kwargs)

            def sdp_bound(self, node, P0, P, *args, **kwargs):
                self.mark("BIQBIN_TEST_RANK1_RANK2_SDP_RETURNS_INVALID_BOUND")
                self.set_sdp_primal_solution(np.eye(P.n, dtype=np.float64))
                return -10

        return Solver

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
                # Deliberately do not call:
                # self.set_sdp_primal_solution(...)
                return 1.0e9

        return Solver

    if case == "heuristic_wrong_length":
        class Solver(BaseSolver):  # type: ignore
            def sdp_bound(self, node, P0, P, *args, **kwargs):
                return self.good_sdp_bound(node, P0, P)

            def heuristic(self, L, *args, **kwargs):
                self.mark("BIQBIN_TEST_HEURISTIC_WRONG_LENGTH")
                P = kwargs["P"]
                return np.zeros(P.n - self._MC_OFFSET - 1, dtype=np.int32)

        return Solver

    if case == "heuristic_non_integer":
        class Solver(BaseSolver):  # type: ignore
            def sdp_bound(self, node, P0, P, *args, **kwargs):
                return self.good_sdp_bound(node, P0, P)

            def heuristic(self, L, *args, **kwargs):  # type: ignore
                self.mark("BIQBIN_TEST_HEURISTIC_NON_INTEGER")
                P = kwargs["P"]
                return np.full(P.n - self._MC_OFFSET, 0.5, dtype=np.float64)

        return Solver

    if case == "heuristic_non_binary01_vector":
        class Solver(BaseSolver):  # type: ignore
            def sdp_bound(self, node, P0, P, *args, **kwargs):
                return self.good_sdp_bound(node, P0, P)

            def heuristic(self, L, *args, **kwargs):
                self.mark("BIQBIN_TEST_HEURISTIC_NON_BINARY01_VECTOR")
                P = kwargs["P"]
                return np.full(P.n - self._MC_OFFSET, 2, dtype=np.int32)

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
