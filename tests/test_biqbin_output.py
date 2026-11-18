import json

import numpy as np
import pytest

from tests.behavior_report import compare_behavior
from tests.solution_checks import assert_binary_vector, assert_cut, load_problem

"""
    Compares Biqbin output with expected output for all Python versions
"""


def test_biqbin_output(problem_instance, subtests, behavior_report):
    """
    Compare Biqbin output with expected output for one instance.
    """

    with open(problem_instance + ".output.json", "r") as f:
        result = json.load(f)

    with open(problem_instance + "-expected_output.json", "r") as f:
        expected_result = json.load(f)

    # Write diagnostics before any correctness assertion, so failures retain them.
    report, changes = compare_behavior(problem_instance, expected_result, result)
    with behavior_report.open("a") as output:
        output.write(report + "\n" + "=" * 80 + "\n\n")
    print(f"Behavior: {changes} changed fields for {problem_instance}; see {behavior_report}")

    problem = load_problem(result)
    adjacency = problem.maxcut_adjacency_matrix * problem.gcd

    # Meta data
    bab_nodes_diff = (
        expected_result["meta_data"]["eval_bab_nodes"] - result["meta_data"]["eval_bab_nodes"]
    )
    time_diff = expected_result["meta_data"]["time"] - result["meta_data"]["time"]

    with subtests.test(
        f"Bab nodes diff (exp - calc) = {bab_nodes_diff} Time diff (exp - calc) = {time_diff:.3f}"
    ):
        # Best way I found to pretty print the bab nodes and time diff
        assert True

    # Check if non-branching instances branched
    if expected_result["meta_data"]["eval_bab_nodes"] == 1:
        with subtests.test("Non-branching instances did not branch"):
            assert bab_nodes_diff == 0, (
                "Bab nodes mismatch!\n",
                f"Got:      {result['meta_data']['eval_bab_nodes']}\n"
                f"Expected: {expected_result['meta_data']['eval_bab_nodes']}\n",
            )

    # --- Check maxcut ---
    with subtests.test("Max-Cut objective value"):
        assert expected_result["maxcut"]["computed_val"] == result["maxcut"]["computed_val"], (
            f"maxcut mismatch!\n"
            f"Got:      {result['maxcut']['computed_val']}\n"
            f"Expected: {expected_result['maxcut']['computed_val']}\n"
            f"Got:      {result['maxcut']['x']}\n"
            f"Expected: {expected_result['maxcut']['x']}"
        )

    with subtests.test("Max-Cut solution"):
        assert_cut(result["maxcut"], adjacency)

    # --- Check qubo if present ---
    if "qubo" in expected_result:
        with subtests.test("QUBO objective value"):
            assert expected_result["qubo"]["computed_val"] == result["qubo"]["computed_val"], (
                f"qubo mismatch!\n"
                f"Got solution: {result['qubo']['computed_val']}\n"
                f"Expected solution: {expected_result['qubo']['computed_val']}\n"
                f"Got solution: {result['qubo']['x']}\n"
                f"Expected solution: {expected_result['qubo']['x']}"
            )
        with subtests.test("QUBO solution vector"):
            qubo = result["qubo"]
            x = assert_binary_vector(qubo["x"], problem.Q.shape[0])
            assert qubo["solution"] == (np.flatnonzero(x) + 1).tolist()
            assert qubo["computed_val"] == pytest.approx(
                x @ problem.Q @ x + problem.offset, rel=0, abs=1e-6
            )
            assert qubo["cardinality"] == int(x.sum())
            assert qubo["minimization"] == problem.is_minimization
            assert qubo["offset"] == problem.offset

    if "bqp" in expected_result:
        with subtests.test("BQP solution"):
            bqp = result["bqp"]
            expected_bqp = expected_result["bqp"]
            assert bqp["feasible_solution"] == expected_bqp["feasible_solution"]
            assert bqp["computed_val"] == expected_bqp["computed_val"]
            if bqp["feasible_solution"]:
                x = assert_binary_vector(bqp["x"], problem.F.shape[0])
                np.testing.assert_array_equal(problem.A @ x, problem.b)
                assert bqp["computed_val"] == pytest.approx(
                    x @ problem.F @ x + problem.c @ x + problem.offset, rel=0, abs=1e-6
                )
            else:
                assert bqp["x"] is None

    expected_root = expected_result["meta_data"]["root_node"]
    computed_root = result["meta_data"]["root_node"]

    # Bounds and heuristic trajectories can vary with MPI scheduling and RNG.
    # Check their mathematical contracts instead of comparing one recorded run.
    with subtests.test("Root node bounds and solution"):
        scale = result["meta_data"]["parameters"]["gcd"]
        optimum = result["maxcut"]["computed_val"] / scale
        bound = computed_root["sdp_value"]
        heuristic = computed_root["heuristic_value"]
        assert np.isfinite(bound)
        assert bound >= optimum - 1e-3
        assert heuristic <= optimum + 1e-6
        assert_cut(
            {"x": computed_root["root_solution"], "computed_val": heuristic},
            adjacency / scale,
        )
        count = computed_root["heuristic_call_count"]
        assert isinstance(count, int) and count >= 0
        if "heuristic_data" in expected_root:
            data = computed_root["heuristic_data"]
            assert len(data) == count
            for entry in data:
                assert np.isfinite(entry["time"]) and entry["time"] >= 0
                assert np.isfinite(entry["value"]) and entry["value"] <= optimum + 1e-6

    if "custom_solver_tests" in expected_result["meta_data"]:
        exp_cs_tests = expected_result["meta_data"]["custom_solver_tests"]
        comp_cs_tests = result["meta_data"]["custom_solver_tests"]

        for key in exp_cs_tests:
            with subtests.test(key):
                if exp_cs_tests[key] != comp_cs_tests[key]:
                    if exp_cs_tests["sdp_calls"] == 0 and key == "heuristic_calls":
                        # 4 test cases include the default SDPBound and custom heuristic
                        # in these cases the heuristic calls are called a indeterminant amount
                        # based on the which worker get's which problem with what RNG
                        assert isinstance(comp_cs_tests[key], int) and comp_cs_tests[key] > 0
                    else:
                        raise ValueError(f"(exp - comp) {exp_cs_tests[key]}-{comp_cs_tests[key]}")
