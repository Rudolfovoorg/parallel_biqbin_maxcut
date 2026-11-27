"""Exercise output naming through real MPI CLI runs and real JSON writers."""

import json
import os
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "script,instance,section",
    [
        ("biqbin_maxcut.py", "tests/rudy/g05_60.0.json", "maxcut"),
        ("biqbin_qubo.py", "tests/custom_solvers_results/small_example_qubo.json", "qubo"),
        ("biqbin_bqp.py", "tests/bqp/test_bqp.data", "bqp"),
    ],
)
def test_cli_overwrite_and_numbered_output(tmp_path, script, instance, section):
    output = tmp_path / "result.json"
    numbered = tmp_path / "result_1.json"
    env = os.environ.copy()
    env.update(
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        GOTO_NUM_THREADS="1",
        OMPI_ALLOW_RUN_AS_ROOT="1",
        OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1",
        OMPI_MCA_rmaps_base_oversubscribe="1",
    )
    command = [
        "timeout",
        "--kill-after=3s",
        "25s",
        shutil.which("mpiexec") or "mpiexec",
        "-n",
        "3",
        sys.executable,
        script,
        instance,
        "-o",
        str(output),
    ]

    def run(*flags):
        result = subprocess.run(
            command + list(flags), env=env, capture_output=True, text=True, timeout=35
        )
        assert result.returncode == 0, result.stdout + result.stderr

    run()
    original = output.read_bytes()
    assert section in json.loads(original)
    run()
    assert output.read_bytes() == original
    assert section in json.loads(numbered.read_text())
    numbered_before = numbered.read_bytes()
    output.write_text("replace this sentinel")
    run("--overwrite")
    assert section in json.loads(output.read_text())
    assert numbered.read_bytes() == numbered_before
    assert sorted(p.name for p in tmp_path.glob("*.json")) == ["result.json", "result_1.json"]
