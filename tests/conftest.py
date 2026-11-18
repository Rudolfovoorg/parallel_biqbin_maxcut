from pathlib import Path

import pytest


def pytest_addoption(parser):
    """
    Adds a custom CLI argument to pytest:
        --instances inst1 inst2 inst3 ...
    """
    parser.addoption(
        "--behavior-report",
        default="test-results/behavior/output.txt",
        help="Write expected-versus-actual solver behavior diagnostics to this file.",
    )
    parser.addoption(
        "--instances",
        nargs="+",
        default=[],
        help="List of problem instances",
    )


def pytest_generate_tests(metafunc):
    """
    Parametrize tests dynamically based on --instances argument.
    """
    if "problem_instance" in metafunc.fixturenames:
        instances = metafunc.config.getoption("--instances")

        if not instances:
            pytest.fail("No instances provided! Use:\n   pytest --instances inst1 inst2 inst3")

        metafunc.parametrize("problem_instance", instances)


@pytest.fixture(scope="session")
def behavior_report(request):
    path = Path(request.config.getoption("--behavior-report"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Solver behavior compared with expected output\n\n")
    return path
