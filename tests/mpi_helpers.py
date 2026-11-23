"""MPI subprocess helpers with bounded execution and saved diagnostics."""

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_mpi(directory, *args, success=True):
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(ROOT),
        OPENBLAS_NUM_THREADS="1",
        GOTO_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        OMPI_ALLOW_RUN_AS_ROOT="1",
        OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1",
        OMPI_MCA_rmaps_base_oversubscribe="1",
    )
    command = [shutil.which("mpiexec") or "mpiexec", "-n", "3", sys.executable, *args]
    process = subprocess.Popen(
        command,
        cwd=directory,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        pytest.fail(f"MPI pipeline hung\n{stdout}\n{stderr}")
    (directory / "stdout.log").write_text(stdout)
    (directory / "stderr.log").write_text(stderr)
    diagnostics = stdout + stderr
    if success:
        assert process.returncode == 0, diagnostics
    else:
        assert process.returncode != 0, diagnostics
    return diagnostics
