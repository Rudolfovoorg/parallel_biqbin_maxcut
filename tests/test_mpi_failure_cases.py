from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


MPIEXEC = os.environ.get("MPIEXEC") or shutil.which(
    "mpirun") or shutil.which("mpiexec")
if MPIEXEC is None:
    raise RuntimeError("mpirun/mpiexec not found")

FAILURE_CASES = {
    # Explicit abort and raw Python callback exceptions.
    "callback_abort_rank0": "BIQBIN_TEST_CALLBACK_ABORT_RANK0",
    "callback_exception_rank0": "BIQBIN_TEST_CALLBACK_EXCEPTION_RANK0",

    # Exception handling at each Python callback boundary.
    "initial_sdp_exception_rank0": "BIQBIN_TEST_INITIAL_SDP_EXCEPTION_RANK0",
    "sdp_exception_rank0": "BIQBIN_TEST_SDP_EXCEPTION_RANK0",
    "initial_heuristic_exception_rank0": "BIQBIN_TEST_INITIAL_HEURISTIC_EXCEPTION_RANK0",
    "heuristic_exception_rank0": "BIQBIN_TEST_HEURISTIC_EXCEPTION_RANK0",

    # Initial SDP return-value validation.
    "initial_sdp_returns_invalid_bound": "BIQBIN_TEST_INITIAL_SDP_RETURNS_INVALID_BOUND",
    "initial_sdp_returns_nan": "BIQBIN_TEST_INITIAL_SDP_RETURNS_NAN",
    "initial_sdp_returns_inf": "BIQBIN_TEST_INITIAL_SDP_RETURNS_INF",
    "initial_sdp_missing_primal": "BIQBIN_TEST_INITIAL_SDP_MISSING_PRIMAL",

    # Non-initial SDP return-value validation.
    "sdp_returns_nan": "BIQBIN_TEST_SDP_RETURNS_NAN",
    "sdp_returns_inf": "BIQBIN_TEST_SDP_RETURNS_INF",
    "sdp_returns_none": "BIQBIN_TEST_SDP_RETURNS_NONE",
    "sdp_returns_string": "BIQBIN_TEST_SDP_RETURNS_STRING",
    "sdp_returns_array": "BIQBIN_TEST_SDP_RETURNS_ARRAY",
    "sdp_rank1_rank2_returns_invalid_bound": "BIQBIN_TEST_RANK1_RANK2_SDP_RETURNS_INVALID_BOUND",

    # SDP primal-solution validation.
    "sdp_wrong_primal_shape": "BIQBIN_TEST_SDP_WRONG_PRIMAL_SHAPE",
    "sdp_missing_primal": "BIQBIN_TEST_SDP_MISSING_PRIMAL",
    "sdp_primal_contains_nan": "BIQBIN_TEST_SDP_PRIMAL_CONTAINS_NAN",
    "sdp_primal_contains_inf": "BIQBIN_TEST_SDP_PRIMAL_CONTAINS_INF",
    "sdp_primal_out_of_range": "BIQBIN_TEST_SDP_PRIMAL_OUT_OF_RANGE",
    "sdp_primal_not_2d": "BIQBIN_TEST_SDP_PRIMAL_NOT_2D",
    "sdp_primal_non_square": "BIQBIN_TEST_SDP_PRIMAL_NON_SQUARE",

    # Heuristic solution validation.
    "heuristic_wrong_length": "BIQBIN_TEST_HEURISTIC_WRONG_LENGTH",
    "heuristic_non_binary01_vector": "BIQBIN_TEST_HEURISTIC_NON_BINARY01_VECTOR",

    # Initial heuristic validation.
    "initial_heuristic_wrong_length": "BIQBIN_TEST_INITIAL_HEURISTIC_WRONG_LENGTH",
    "initial_heuristic_non_binary01_vector": "BIQBIN_TEST_INITIAL_HEURISTIC_NON_BINARY01_VECTOR",

    # Native arrays are intended to be read-only.
    "callback_mutates_problem_matrix": "BIQBIN_TEST_MUTATE_PROBLEM_MATRIX",
    "callback_mutates_node_solution": "BIQBIN_TEST_MUTATE_NODE_SOLUTION",

    # SDP returns a bound that is proved wrong by the heuristic
    "sdp_bound_below_heuristic": "BIQBIN_TEST_SDP_BOUND_BELOW_HEURISTIC",
    "initial_sdp_bound_below_heuristic": "BIQBIN_TEST_INITIAL_SDP_BOUND_BELOW_HEURISTIC",
}


def run_case(case: str, tmp_path: Path) -> tuple[subprocess.CompletedProcess[str], Path]:
    log_dir = tmp_path / case
    log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONFAULTHANDLER"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["GOTO_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"

    # Useful when running OpenMPI in Docker/root-like environments.
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")

    cmd = [
        MPIEXEC,
        "-n",
        "3",
        sys.executable,
        "-m",
        "tests.mpi_failure_case_runner",
        "--case",
        case,
        "--log-dir",
        str(log_dir),
    ]

    cp = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )

    return cp, log_dir


def marker_files(log_dir: Path, marker: str) -> list[Path]:
    return sorted(log_dir.glob(f"*.{marker}.marker"))


@pytest.mark.parametrize("case, marker", FAILURE_CASES.items())
def test_mpi_failure_case(case: str, marker: str, tmp_path: Path) -> None:
    cp, log_dir = run_case(case, tmp_path)

    markers = marker_files(log_dir, marker)

    diagnostics = (f"case: {case}\n"
                   f"marker: {marker}\n"
                   f"returncode: {cp.returncode}\n"
                   f"log_dir: {log_dir}\n"
                   f"marker_files: {[str(path) for path in markers]}\n\n"
                   f"stdout:\n{cp.stdout}\n\n"
                   f"stderr:\n{cp.stderr}\n"
                   )

    assert cp.returncode != 0,  diagnostics
    assert markers, diagnostics
