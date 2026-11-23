"""Run the public pipeline under MPI, away from the source checkout."""

import json
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


def qubo_input(directory):
    path = directory / "input.json"
    path.write_text(
        json.dumps(
            {
                "qubo": {
                    "shape": [2, 2],
                    "nnz": 4,
                    "row": [0, 0, 1, 1],
                    "col": [0, 1, 0, 1],
                    "data": [-1, 3, 3, -1],
                },
                "offset": 10,
                "is_minimization": True,
            }
        )
    )
    return path


@pytest.mark.parametrize(
    "kind,instance",
    [
        ("maxcut", "tests/rudy/g05_60.0.json"),
        ("qubo", "tests/custom_solvers_results/small_example_qubo.json"),
        ("bqp", "tests/bqp/test_bqp.data"),
    ],
)
def test_module_and_legacy_entrypoints_work_without_checkout_params(tmp_path, kind, instance):
    assert not (tmp_path / "params").exists()
    actual = []
    for entry, args in [
        ("module", ["-m", "biqbin", kind]),
        ("legacy", [str(ROOT / f"biqbin_{kind}.py")]),
    ]:
        output = tmp_path / f"{entry}.json"
        run_mpi(tmp_path, *args, str(ROOT / instance), "-o", str(output))
        result = json.loads(output.read_text())
        actual.append(result[kind]["computed_val"])
        assert result["meta_data"]["solver_parameters"]["time_limit"] == 0
    assert actual[0] == actual[1]


def test_file_parameter_time_limit_is_not_replaced_by_absent_cli_flag(tmp_path):
    instance = qubo_input(tmp_path)
    params = tmp_path / "legacy.params"
    params.write_text("time_limit = 120\n")
    output = tmp_path / "result.json"
    run_mpi(tmp_path, "-m", "biqbin", "qubo", str(instance), "-p", str(params), "-o", str(output))
    result = json.loads(output.read_text())
    assert result["qubo"]["computed_val"] == 9
    assert result["meta_data"]["solver_parameters"]["time_limit"] == 120


def test_config_relative_paths_and_explicit_overrides(tmp_path):
    config_dir = tmp_path / "configuration"
    config_dir.mkdir()
    qubo_input(config_dir)
    config = config_dir / "run.toml"
    config.write_text(
        'kind = "qubo"\n'
        '[reader]\npath = "input.json"\n'
        "[problem]\noptimize = true\n"
        "[solver]\ntime_limit = 60\nmax_bundle_iter = 30\n"
        '[writer]\npath = "result.json"\noverwrite = true\nwith_maxcut_solution = true\n'
        "[run]\ncollect_root_data = true\n"
    )
    legacy = tmp_path / "legacy.params"
    legacy.write_text("time_limit = 120\nmax_bundle_iter = 40\n")
    sentinel = config_dir / "result.json"
    sentinel.write_text("keep existing result")
    run_mpi(
        tmp_path,
        "-m",
        "biqbin",
        "qubo",
        "--config",
        str(config),
        "-p",
        str(legacy),
        "--param",
        "max_bundle_iter=45",
        "-t",
        "0",
        "--no-optimize",
        "--no-overwrite",
        "--no-collect-root-data",
        "--no-maxcut-solution",
    )
    assert sentinel.read_text() == "keep existing result"
    result = json.loads((config_dir / "result_1.json").read_text())
    assert result["qubo"]["computed_val"] == 9
    assert "maxcut" not in result
    meta = result["meta_data"]
    assert meta["solver_parameters"]["time_limit"] == 0
    assert meta["solver_parameters"]["max_bundle_iter"] == 45
    assert meta["runtime_config"]["problem"]["optimize"] is False
    assert meta["runtime_config"]["run"]["collect_root_data"] is False


@pytest.mark.parametrize(
    "parameter", ["unknown_parameter=1", "time_limit=-1", "max_bundle_iter=bad"]
)
def test_invalid_parameters_fail_without_hanging(tmp_path, parameter):
    instance = qubo_input(tmp_path)
    run_mpi(tmp_path, "-m", "biqbin", "qubo", str(instance), "--param", parameter, success=False)
    assert not list(tmp_path.glob("*.output*"))


def test_missing_input_fails_without_hanging(tmp_path):
    run_mpi(tmp_path, "-m", "biqbin", "qubo", "missing.json", success=False)


def test_unknown_config_section_fails_without_hanging(tmp_path):
    instance = qubo_input(tmp_path)
    config = tmp_path / "bad.toml"
    config.write_text('[solvre]\ntime_limit = 120\n')
    diagnostics = run_mpi(
        tmp_path, "-m", "biqbin", "qubo", str(instance), "--config", str(config), success=False
    )
    assert "solvre" in diagnostics


def test_missing_initial_solution_fails_without_hanging(tmp_path):
    instance = qubo_input(tmp_path)
    run_mpi(
        tmp_path, "-m", "biqbin", "qubo", str(instance), "-s", "missing-initial.json", success=False
    )
    assert not list(tmp_path.glob("*.output*"))


def test_worker_factory_failure_is_collective(tmp_path):
    instance = qubo_input(tmp_path)
    script = tmp_path / "custom.py"
    script.write_text(
        "from biqbin import QUBOSolver, get_rank\n"
        "from biqbin.runtime import run\n"
        "def factory(problem, **kwargs):\n"
        "    if get_rank() == 1:\n"
        "        raise ValueError('worker factory failure')\n"
        "    return QUBOSolver(problem, **kwargs)\n"
        f"run({{'kind': 'qubo', 'reader': {{'path': {str(instance)!r}}}}}, solver_factory=factory)\n"
    )
    diagnostics = run_mpi(tmp_path, str(script), success=False)
    assert "worker factory failure" in diagnostics
