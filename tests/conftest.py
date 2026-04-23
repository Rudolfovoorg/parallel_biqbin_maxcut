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
    parser.addoption(
        "--without-sol-vector",
        action="store_true",
        help="Skip checking the solution vector for SA heurist, it is too random",
    )


def pytest_generate_tests(metafunc):
    """
    Parametrize tests dynamically based on --instances argument.
    """
    if "problem_instance" in metafunc.fixturenames:
        instances = metafunc.config.getoption("--instances")

        if not instances:
            pytest.fail(
                "No instances provided! Use:\n"
                "   pytest --instances inst1 inst2 inst3"
            )

        metafunc.parametrize("problem_instance", instances)
