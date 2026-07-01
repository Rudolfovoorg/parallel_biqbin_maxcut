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
    "callback_exception_rank0": "BIQBIN_TEST_CALLBACK_EXCEPTION_RANK0",
    "sdp_returns_nan": "BIQBIN_TEST_SDP_RETURNS_NAN",
    "sdp_returns_inf": "BIQBIN_TEST_SDP_RETURNS_INF",
    "sdp_wrong_primal_shape": "BIQBIN_TEST_SDP_WRONG_PRIMAL_SHAPE",
    "sdp_missing_primal": "BIQBIN_TEST_SDP_MISSING_PRIMAL",
    "heuristic_wrong_length": "BIQBIN_TEST_HEURISTIC_WRONG_LENGTH",
    "heuristic_non_integer": "BIQBIN_TEST_HEURISTIC_NON_INTEGER",
    "heuristic_non_binary01_vector": "BIQBIN_TEST_HEURISTIC_NON_BINARY01_VECTOR",
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
        timeout=30,
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
