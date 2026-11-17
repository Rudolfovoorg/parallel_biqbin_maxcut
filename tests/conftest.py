import pytest


def pytest_addoption(parser):
    """
    Adds a custom CLI argument to pytest:
        --instances inst1 inst2 inst3 ...
    """
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
