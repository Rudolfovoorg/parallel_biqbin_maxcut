from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from biqbin import ProblemMaxCut, ProblemQubo, SolutionMaxCut, SolutionQubo
from biqbin_bqp import SolutionBQP


def make_solution(kind, time_limit):
    raw = {
        "maxcut": {"computed_val": 9, "solution": [], "x": [0, 0, 0]},
        "meta_data": {
            "instance": "example.json", "time": 2.987,
            "time_limit_reached": time_limit, "eval_bab_nodes": 17,
            "heuristic_call_count": 126, "num_workers_used": 7,
            "parameters": {"optimized": False, "gcd": 1},
        },
    }
    if kind == "Max-Cut":
        problem = ProblemMaxCut(np.zeros((3, 3)), "example.json")
        solution = SolutionMaxCut(raw, problem)
    elif kind == "QUBO":
        problem = ProblemQubo(np.zeros((2, 2)), 5, "example.json", True)
        solution = SolutionQubo(raw, problem)
    else:
        problem = SimpleNamespace(
            F=np.zeros((2, 2)), c=np.zeros(2),
            A=np.array([[1, 0]]), b=np.array([1]),
            rho=1.0, const_value=20 if kind == "BQP infeasible" else 9,
            penalty=3, offset=7, optimize_mc_adj_matrix=False,
        )
        solution = SolutionBQP(raw, problem)
    return solution, raw


@pytest.mark.parametrize("kind", ["Max-Cut", "QUBO", "BQP", "BQP infeasible"])
@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize("time_limit", [False, True])
def test_shared_summary(kind, verbose, time_limit, capsys):
    solution, raw = make_solution(kind, time_limit)
    before = deepcopy(raw)
    result_before = deepcopy(solution.solution)
    text = solution.format_summary(verbose)
    assert "\n\n---" not in text
    header = text.split("\n---", 1)[0]
    assert header.splitlines()[0] == type(solution).__name__
    fields = dict(line.split(" = ", 1) for line in header.splitlines()[1:])
    fields = {key.strip(): value for key, value in fields.items()}
    assert fields == {
        "Problem name": "example.json", "Compute time": "2.99 seconds",
        "Time limit reached": str(time_limit), "B&B nodes evaluated": "17",
        "Heuristic calls": "126", "Worker processes used": "7",
        "Optimized input": "False", "GCD": "1",
    }
    assert text.count("Problem name") == 1
    title = "BQP" if kind.startswith("BQP") else kind
    assert f"--- {title} ---" in text
    assert ("Underlying Max-Cut" in text) == (verbose and kind != "Max-Cut")
    assert ("BQP transformation" in text) == (verbose and kind.startswith("BQP"))
    assert ("Penalty" in text) == (verbose and kind.startswith("BQP"))
    if kind.startswith("BQP"):
        assert f"Feasible solution     = {kind == 'BQP'}" in text
        if kind == "BQP infeasible":
            assert "Computed value        = None" in text
            assert "x                     = None" in text
        else:
            assert "Computed value        = 7" in text
    elif kind == "QUBO":
        assert "Computed value        = 5.0" in text
        assert "Minimization          = True" in text
    solution.print_computed_solution(verbose)
    assert capsys.readouterr().out == text + "\n"
    assert str(solution) == solution.format_summary(False)
    assert raw == before
    assert solution.solution == result_before


@pytest.mark.parametrize("kind", ["Max-Cut", "QUBO", "BQP"])
@pytest.mark.parametrize("verbose", [False, True])
def test_problem_summary(kind, verbose, monkeypatch):
    if kind == "Max-Cut":
        problem = ProblemMaxCut(np.array([[0, 2], [2, 0]]), "example.json", True)
    elif kind == "QUBO":
        problem = ProblemQubo(np.array([[2, 0], [0, 4]]), 3, "example.json", False, True)
    else:
        import biqbin_bqp
        monkeypatch.setattr(biqbin_bqp, "get_rank", lambda: 0)
        monkeypatch.setattr(biqbin_bqp, "interior_point_method_maxcut", lambda C: (0.0, None))
        problem = biqbin_bqp.ProblemBQP(
            np.zeros((2, 2)), np.zeros(2), np.array([[1, 1]]),
            np.array([1]), "example.json", offset=3, optimize_input=True,
        )
    matrix_before = problem.maxcut_adjacency_matrix.copy()
    text = problem.format_summary(verbose)
    assert text.splitlines()[0] == type(problem).__name__
    assert text.count("Problem name") == 1
    assert "Problem name          = example.json" in text
    assert "Optimized input       = True" in text
    assert f"GCD                   = {problem.gcd}" in text
    assert f"--- {kind} ---" in text
    assert "\n\n---" not in text
    assert ("Underlying Max-Cut" in text) == (verbose and kind != "Max-Cut")
    assert ("BQP transformation" in text) == (verbose and kind == "BQP")
    if kind == "QUBO":
        assert "Minimization          = False" in text
        assert "Offset                = 3" in text
        assert "Q = (2, 2)" in text
    elif kind == "BQP":
        assert "Equality constraints  = 1" in text
        for label in ("F = (2, 2)", "c = (2,)", "A = (1, 2)", "b = (1,)"):
            assert label in text
    assert str(problem) == problem.format_summary(False)
    assert repr(problem) == str(problem)
    np.testing.assert_array_equal(problem.maxcut_adjacency_matrix, matrix_before)


@pytest.mark.parametrize("kind", ["Max-Cut", "QUBO", "BQP"])
def test_log_record_starts_with_class_heading(kind):
    import io
    import logging

    solution, _ = make_solution(kind, False)
    objects = [solution]
    if kind != "BQP":
        objects.append(solution.problem)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    test_logger = logging.Logger("summary-test", logging.INFO)
    test_logger.addHandler(handler)
    for obj in objects:
        stream.seek(0)
        stream.truncate()
        test_logger.info("%s", obj)
        assert stream.getvalue() == "INFO: " + obj.format_summary() + "\n"
        assert stream.getvalue().splitlines()[0] == "INFO: " + type(obj).__name__
