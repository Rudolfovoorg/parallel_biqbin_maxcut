import ctypes
import importlib
import io
import os
from pathlib import Path
import shutil
import signal
import logging
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import biqbin
import biqbin.biqbin_base as base
from biqbin.argparsers import ArgParserMaxCut

log_module = importlib.import_module("biqbin.logging")


@pytest.fixture(autouse=True)
def restore_logger():
    logger = log_module.logger
    handlers = logger.handlers[:]
    owned = log_module._configured_handlers[:]
    level, propagate = logger.level, logger.propagate
    yield
    for handler in logger.handlers[:]:
        if handler not in handlers:
            logger.removeHandler(handler)
            handler.close()
    logger.handlers[:] = handlers
    log_module._configured_handlers[:] = owned
    logger.setLevel(level)
    logger.propagate = propagate


def test_import_does_not_configure_root():
    subprocess.run([sys.executable, "-c", """
import logging
root = logging.getLogger()
before = (root.handlers[:], root.level)
import biqbin
assert (root.handlers, root.level) == before
assert biqbin.logger is logging.getLogger('biqbin')
"""], check=True)


def test_reconfiguration_preserves_external_handlers(tmp_path, capsys):
    root = logging.getLogger()
    root_state = (root.handlers[:], root.level)
    stream = io.StringIO()
    external = logging.StreamHandler(stream)
    biqbin.logger.addHandler(external)
    path = tmp_path / "solver.log"
    for _ in range(2):
        biqbin.configure_logging("DEBUG", filename=path, format="%(levelname)s:%(message)s")
    biqbin.logger.debug("value=%s", 42)
    assert capsys.readouterr().err == "DEBUG:value=42\n"
    assert path.read_text() == "DEBUG:value=42\n"
    assert stream.getvalue() == "value=42\n"
    assert (root.handlers, root.level) == root_state
    biqbin.configure_logging("WARNING")
    biqbin.logger.warning("console only")
    assert path.read_text() == "DEBUG:value=42\n"


def test_cli_verbosity_resets_and_filters(capsys):
    parser = ArgParserMaxCut()
    for flags, expected in [(["-vv"], logging.DEBUG), (["-v"], logging.INFO), ([], logging.WARNING)]:
        parser.parse_args(["example.json", *flags])
        assert biqbin.logger.level == expected
    biqbin.logger.debug("hidden")
    biqbin.logger.warning("visible")
    output = capsys.readouterr().err
    assert "hidden" not in output
    assert output.count("visible") == 1


def test_native_bridge_uses_shared_configuration(capsys):
    native = importlib.import_module("biqbin.biqbin_module")
    bridge = ctypes.PyDLL(native.__file__).biqbin_log_write
    bridge.restype = None
    bridge.argtypes = [ctypes.c_int, ctypes.c_char_p]
    biqbin.configure_logging("WARNING", format="%(levelname)s:%(name)s:%(message)s")
    bridge(logging.WARNING, b"native record")
    bridge(logging.INFO, b"filtered record")
    assert capsys.readouterr().err == "WARNING:biqbin:native record\n"


@pytest.mark.parametrize("level", [logging.WARNING, logging.DEBUG])
def test_heuristic_logging_does_not_run_extra_computation(monkeypatch, level):
    biqbin.logger.setLevel(level)
    gw = Mock(side_effect=AssertionError("Logging must not run another heuristic"))
    update = Mock()
    monkeypatch.setattr(base, "goemans_williamson_heuristic", gw)
    monkeypatch.setattr(base, "update_mc_lower_bound_solution", update)
    solver = SimpleNamespace(
        heuristic_call_count=0,
        _heuristic_fn=Mock(return_value=np.array([1, 0])),
        _check_solution_validity=Mock(),
        _evaluate_solution=Mock(return_value=17.0),
        _MC_OFFSET=1,
        _primal_solution_set=True,
    )
    node = SimpleNamespace(xfixed=np.array([0, 0]), sol=SimpleNamespace(x=np.array([0, 0])))
    problem = SimpleNamespace(n=3, L=np.zeros((3, 3)))
    result = base.MaxCutSolver._call_heuristic(solver, node, problem, problem)
    assert result == 17.0
    assert solver.heuristic_call_count == 1
    solver._heuristic_fn.assert_called_once()
    solver._evaluate_solution.assert_called_once()
    update.assert_called_once()
    np.testing.assert_array_equal(update.call_args.args[0], [1, 0])
    gw.assert_not_called()


@pytest.mark.parametrize("flags, expected", [
    ([], None),
    (["--log-dir"], "example.logs"),
    (["--log-dir", "custom/logs"], "custom/logs"),
])
def test_cli_log_directory(flags, expected):
    args = ArgParserMaxCut().parse_args(["inputs/example.json", *flags])
    assert args.log_dir == expected
    assert not any(isinstance(h, logging.FileHandler) for h in biqbin.logger.handlers)


def test_rank_files_are_separate_and_append(tmp_path, capsys):
    directory = tmp_path / "nested" / "logs"
    biqbin.configure_logging("INFO")
    for rank, message in [(0, "first rank"), (1, "second rank"), (0, "first again")]:
        log_module.configure_rank_logging(directory, rank)
        biqbin.logger.info(message)
    first = (directory / "rank-00000.log").read_text()
    second = (directory / "rank-00001.log").read_text()
    assert first.count("first rank") == 1
    assert first.count("first again") == 1
    assert "second rank" not in first
    assert second.count("second rank") == 1
    assert "first" not in second
    assert capsys.readouterr().err.count("first again") == 1


@pytest.mark.parametrize("failure", ["directory", "file"])
def test_rank_logging_failure_aborts_mpi_job(tmp_path, failure):
    mpiexec = shutil.which("mpiexec")
    if mpiexec is None:
        pytest.skip("MPI launcher unavailable")
    root = Path(__file__).resolve().parents[1]
    directory = tmp_path / "logs"
    directory.mkdir()
    if failure == "file":
        (directory / "rank-00001.log").mkdir()
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    code = """
import sys
from biqbin import init, get_rank
from biqbin.logging import configure_rank_logging
from biqbin.biqbin_module import reduce_sum_mpi
init()
rank = get_rank()
directory = sys.argv[2] if rank == 1 and sys.argv[3] == 'directory' else sys.argv[1]
configure_rank_logging(directory, rank)
reduce_sum_mpi(1)
"""
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
    process = subprocess.Popen(
        [mpiexec, "-n", "3", sys.executable, "-c", code,
         str(directory), str(blocked), failure],
        cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        pytest.fail("Log setup failure left MPI peers hanging")
    assert process.returncode != 0, stdout + stderr
    assert "BiqBin rank 1: cannot configure log directory" in stderr
    assert "MPI_ABORT" in stderr.upper()


@pytest.mark.parametrize("value, valid", [
    ("0", True), ("10", True), ("25", True),
    ("debug", True), ("WaRnInG", True), ("TRACE", True), ("FATAL", True),
    ("00", False), ("010", False), ("08", False), ("0x10", False),
    ("-1", False), ("1.5", False), ("unknown", False),
])
def test_make_log_level_validation(value, valid):
    if shutil.which("make") is None:
        pytest.skip("make unavailable")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["make", "-n", "-B", f"LOG_LEVEL={value}", "build/wrapper/bounding.o"],
        cwd=root, text=True, capture_output=True, timeout=20,
    )
    assert (result.returncode == 0) == valid, result.stdout + result.stderr
    if not valid:
        assert "Invalid LOG_LEVEL" in result.stderr


def test_default_format_includes_timestamp_in_console_and_file(tmp_path, capsys):
    import re

    path = tmp_path / "timestamp.log"
    biqbin.configure_logging("INFO", filename=path)
    biqbin.logger.info("timestamp check")
    output = capsys.readouterr().err
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}:INFO:biqbin:pid=\d+:timestamp check\n",
        output,
    )
    assert path.read_text() == output
