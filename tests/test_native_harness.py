"""The shell harness must not turn missing output or solver failures into passes."""

import subprocess
from pathlib import Path

import pytest

HARNESS = Path(__file__).with_name("test.sh")
RESULT = "Maximum value = 3\nSolution = ( 1 )\n"


@pytest.mark.parametrize(
    "output,status,expected,success",
    [
        (RESULT, 0, RESULT, True),
        (RESULT.replace("( 1 )", "( 2 )"), 0, RESULT, True),
        (RESULT.replace("( 1 )", "( 1 2 )"), 0, RESULT, False),
        (RESULT.replace("( 1 )", "( 3 )"), 0, RESULT, False),
        (RESULT.replace("( 1 )", "( 1 1 )"), 0, RESULT, False),
        (RESULT, 7, RESULT, False),
        ("", 0, "", False),
        ("Maximum value = 3\n", 0, RESULT, False),
        (RESULT + RESULT, 0, RESULT, False),
        (RESULT.replace("= 3", "= 4"), 0, RESULT, False),
    ],
)
def test_shell_harness(tmp_path, output, status, expected, success):
    solver = tmp_path / "solver.sh"
    solver.write_text(f"#!/bin/bash\nprintf '%s' '{output}'\nexit {status}\n")
    reference = tmp_path / "expected output"
    reference.write_text(expected)
    graph = tmp_path / "graph"
    graph.write_text("2 1\n1 2 3\n")
    result = subprocess.run(
        [
            "bash",
            str(HARNESS),
            f"bash {solver}",
            str(graph),
            str(reference),
            "unused params",
        ],
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    if status:
        assert result.returncode == status
