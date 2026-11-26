from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

MPIEXEC = os.environ.get("MPIEXEC") or shutil.which("mpirun") or shutil.which("mpiexec")
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


def run_case(
    case: str, tmp_path: Path, timeout: float = 60
) -> tuple[subprocess.CompletedProcess[str], Path]:
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

    process = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        start_new_session=True,
    )

    stdout = stderr = ""
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Kill the launcher and its local ranks; timeout is never an expected abort.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
        pytest.fail(
            f"MPI case {case} hung for {timeout} seconds\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )
    finally:
        # Preserve diagnostics even for expected failures.
        (log_dir / "stdout.log").write_text(stdout)
        (log_dir / "stderr.log").write_text(stderr)

    return subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr), log_dir


def marker_files(log_dir: Path, marker: str) -> list[Path]:
    return sorted(log_dir.glob(f"*.{marker}.marker"))


@pytest.mark.parametrize("case, marker", FAILURE_CASES.items())
def test_mpi_failure_case(case: str, marker: str, tmp_path: Path) -> None:
    cp, log_dir = run_case(case, tmp_path)

    markers = marker_files(log_dir, marker)

    diagnostics = (
        f"case: {case}\n"
        f"marker: {marker}\n"
        f"returncode: {cp.returncode}\n"
        f"log_dir: {log_dir}\n"
        f"marker_files: {[str(path) for path in markers]}\n\n"
        f"stdout:\n{cp.stdout}\n\n"
        f"stderr:\n{cp.stderr}\n"
    )

    assert cp.returncode != 0, diagnostics
    assert markers, diagnostics


@pytest.mark.parametrize(
    "case",
    [
        "repeat_same_solver",
        "repeat_new_solver",
        "repeat_maxcut_solver",
        "repeat_bqp_solver",
        "repeat_native_run",
        "repeat_rank0_only",
    ],
)
def test_second_solve_aborts_with_clear_error(case, tmp_path):
    cp, log_dir = run_case(case, tmp_path, timeout=30)
    diagnostics = f"returncode: {cp.returncode}\nstdout:\n{cp.stdout}\nstderr:\n{cp.stderr}"

    assert len(marker_files(log_dir, "BIQBIN_TEST_FIRST_SOLVE_COMPLETED")) == 3, diagnostics
    assert cp.returncode != 0, diagnostics
    assert cp.returncode not in (-signal.SIGSEGV, 128 + signal.SIGSEGV), diagnostics
    assert (
        "Fatal error: BiqBin currently supports only one solve per process. "
        "Start a new MPI job to solve another problem."
    ) in cp.stderr, diagnostics
    assert "UNEXPECTED_SUCCESS" not in cp.stdout, diagnostics


def test_timeout_is_failure_and_stops_process_group(tmp_path, monkeypatch):
    """A hanging launcher and child must be killed, not accepted as an MPI abort."""
    launcher = tmp_path / "hanging-mpi"
    launcher.write_text(
        f"#!{sys.executable}\n"
        "import os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "directory = Path(sys.argv[sys.argv.index('--log-dir') + 1])\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "(directory / 'pids').write_text(f'{os.getpid()} {child.pid}')\n"
        "print('launcher started', flush=True)\n"
        "print('diagnostic stderr', file=sys.stderr, flush=True)\n"
        "time.sleep(60)\n"
    )
    launcher.chmod(0o755)
    monkeypatch.setattr(sys.modules[__name__], "MPIEXEC", str(launcher))
    with pytest.raises(pytest.fail.Exception, match="hung for"):
        run_case("hang", tmp_path, timeout=2)

    directory = tmp_path / "hang"
    assert "launcher started" in (directory / "stdout.log").read_text()
    assert "diagnostic stderr" in (directory / "stderr.log").read_text()
    for pid in (directory / "pids").read_text().split():
        status = Path(f"/proc/{pid}/stat")
        try:
            state = status.read_text().rsplit(") ", 1)[1][0]
        except FileNotFoundError:
            continue
        assert state == "Z", f"Process {pid} still running after timeout: {state}"


def test_compute_uses_its_own_callbacks(tmp_path):
    cp, log_dir = run_case("callback_ownership", tmp_path, timeout=30)
    diagnostics = f"returncode: {cp.returncode}\nstdout:\n{cp.stdout}\nstderr:\n{cp.stderr}"
    assert cp.returncode == 0, diagnostics
    assert len(marker_files(log_dir, "BIQBIN_TEST_CALLBACK_OWNER_PASSED")) == 3, diagnostics
