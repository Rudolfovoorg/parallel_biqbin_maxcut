"""Run the real Makefile recipes with small stand-ins for expensive commands."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

MAKEFILE = Path(__file__).resolve().parents[1] / "Makefile"


def recipe(target):
    lines = MAKEFILE.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(target + ":"))
    end = start + 1
    while end < len(lines):
        line = lines[end]
        if line and not line.startswith(("\t", target + ":")):
            break
        end += 1
    return "\n".join(lines[start:end]) + "\n"


@pytest.fixture
def commands(tmp_path):
    stub = tmp_path / "command.py"
    stub.write_text(
        r"""import os, sys
from pathlib import Path
args = sys.argv[1:]
mode = os.environ.get('FAIL', '')
if args[0] == 'make':
    if mode != 'missing':
        data = Path(os.environ['COVERAGE_FILE'])
        data.touch()
        Path(str(data) + '.rank').touch()
    sys.exit(1 if mode == 'tests' else 0)
if args[:2] == ['-m', 'coverage']:
    command = args[2]
    with open('calls', 'a') as f:
        f.write(command + '\n')
    sys.exit(1 if mode == command else 0)
if args[:2] == ['-m', 'tests.biqbin_custom_solvers']:
    case = args[args.index('--test-case') + 1]
    with open('cases', 'a') as f:
        f.write(case + '\n')
    sys.exit(1 if mode == case else 0)
assert args[:2] == ['-m', 'pytest']
instances = args[args.index('--instances') + 1:]
assert instances == [f'tests/custom_solvers_results/small_example_qubo_test_case_{i}.json' for i in range(15)]
Path('validated').touch()
"""
    )
    return f"{sys.executable} {stub}"


def run_make(tmp_path, fail=""):
    env = os.environ.copy()
    for key in ("MAKEFLAGS", "MFLAGS", "MAKELEVEL"):
        env.pop(key, None)
    env["FAIL"] = fail
    return subprocess.run(
        ["make"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15
    )


@pytest.mark.parametrize("fail", ["", "tests", "missing", "combine", "report", "xml", "html"])
def test_coverage_exit_status(tmp_path, commands, fail):
    (tmp_path / "Makefile").write_text(
        f"PYTHON = {commands}\nMAKE = {commands} make\n"
        "COVERAGE_DIR = artifacts/coverage\nTEST_RESULTS_DIR = artifacts/tests\n"
        + recipe("coverage")
    )
    result = run_make(tmp_path, fail)
    assert (result.returncode != 0) == bool(fail), result.stdout + result.stderr
    calls = (tmp_path / "calls").read_text().splitlines()
    if fail == "missing":
        assert "No combined coverage database" in result.stderr
        assert calls == ["erase"]
    else:
        # Even after a test/combine/report error, attempt the remaining reports.
        assert calls == ["erase", "combine", "report", "xml", "html"]


@pytest.mark.parametrize("fail", ["", "0", "7", "14"])
def test_custom_solver_loop_stops_on_failure(tmp_path, commands, fail):
    (tmp_path / "Makefile").write_text(
        f"PYTHON = {commands}\nMPI_PYTHON = {commands}\n"
        + recipe("test-modular-combos").replace(": clean-output", ":")
    )
    result = run_make(tmp_path, fail)
    assert (result.returncode != 0) == bool(fail), result.stdout + result.stderr
    assert (tmp_path / "cases").read_text().splitlines() == [
        str(i) for i in range(int(fail) + 1 if fail else 15)
    ]
    assert (tmp_path / "validated").exists() == (not fail)
