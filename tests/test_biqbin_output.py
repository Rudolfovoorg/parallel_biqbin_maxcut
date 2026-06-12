import json
import pytest

"""
    Compares Biqbin output with expected output for all Python versions
"""


def test_biqbin_output(problem_instance, request, subtests):
    """
    Compare Biqbin output with expected output for one instance.
    """

    with open(problem_instance + '.output.json', 'r') as f:
        result = json.load(f)

    with open(problem_instance + '-expected_output.json', 'r') as f:
        expected_result = json.load(f)

    # heuristic with SA is too random to check the exact solution vector
    without_sol_vec = request.config.getoption('--without-sol-vector')

    # Meta data
    bab_nodes_diff = expected_result["meta_data"]["eval_bab_nodes"] - \
        result["meta_data"]["eval_bab_nodes"]
    time_diff = expected_result["meta_data"]["time"] - \
        result["meta_data"]["time"]

    with subtests.test(f'Bab nodes diff (exp - calc) = {bab_nodes_diff} Time diff (exp - calc) = {time_diff:.3f}'):
        # Best way I found to pretty print the bab nodes and time diff
        assert True

    # Check if non-branching instances branched
    if expected_result["meta_data"]["eval_bab_nodes"] == 1:
        with subtests.test('Non-branching instances did not branch'):
            assert bab_nodes_diff == 0, (
                'Bab nodes mismatch!\n',
                f'Got:      {result["meta_data"]["eval_bab_nodes"]}\n'
                f'Expected: {expected_result["meta_data"]["eval_bab_nodes"]}\n'
            )

    # --- Check maxcut ---
    with subtests.test('Max-Cut objective value'):
        assert expected_result["maxcut"]["computed_val"] == result["maxcut"]["computed_val"], (
            f'maxcut mismatch!\n'
            f'Got:      {result["maxcut"]["computed_val"]}\n'
            f'Expected: {expected_result["maxcut"]["computed_val"]}\n'
            f'Got:      {result["maxcut"]["x"]}\n'
            f'Expected: {expected_result["maxcut"]["x"]}'
        )

    with subtests.test('Max-Cut solution'):
        if expected_result["maxcut"]["x"] != result["maxcut"]["x"]:
            pytest.xfail(
                f'maxcut mismatch!\n'
                f'Got:      {result["maxcut"]["x"]}\n'
                f'Expected: {expected_result["maxcut"]["x"]}'
            )
        if expected_result["maxcut"]["solution"] != result["maxcut"]["solution"]:
            pytest.xfail(
                f'maxcut mismatch!\n'
                f'Got:      {result["maxcut"]["solution"]}\n'
                f'Expected: {expected_result["maxcut"]["solution"]}'
            )

    # --- Check qubo if present ---
    if 'qubo' in expected_result:
        with subtests.test('QUBO objective value'):
            assert expected_result["qubo"]["computed_val"] == result["qubo"]["computed_val"], (
                f'qubo mismatch!\n'
                f'Got solution: {result["qubo"]["computed_val"]}\n'
                f'Expected solution: {expected_result["qubo"]["computed_val"]}\n'
                f'Got solution: {result["qubo"]["x"]}\n'
                f'Expected solution: {expected_result["qubo"]["x"]}'
            )
        with subtests.test('QUBO solution vector'):
            if expected_result["qubo"]["x"] != result["qubo"]["x"]:
                pytest.xfail(
                    f'qubo mismatch!\n'
                    f'Got solution: {result["qubo"]["x"]}\n'
                    f'Expected solution: {expected_result["qubo"]["x"]}'
                )
            if expected_result["qubo"]["solution"] != result["qubo"]["solution"]:
                pytest.xfail(
                    f'qubo mismatch!\n'
                    f'Got solution: {result["qubo"]["solution"]}\n'
                    f'Expected solution: {expected_result["qubo"]["solution"]}'
                )

    # --- Check bqp if present ---
    if 'bqp' in expected_result:
        with subtests.test('BQP solution'):
            assert expected_result["bqp"] == result["bqp"], (
                f'bqp mismatch!\n'
                f'Got: {result["bqp"]}\n'
                f'Expected: {expected_result["bqp"]}'
            )

    expected_root = expected_result["meta_data"]["root_node"]
    computed_root = result["meta_data"]["root_node"]

    # What should the tolerance of this be?
    with subtests.test('Root node sdp_value'):
        if abs(expected_root["sdp_value"] - computed_root["sdp_value"]) > 0.001:
            pytest.xfail(
                f'root sdp_value mismatch!'
                f'Got:      {computed_root["sdp_value"]} '
                f'Expected: {expected_root["sdp_value"]}'
            )

    with subtests.test('Root node heuristic_value'):
        assert expected_root["heuristic_value"] == computed_root["heuristic_value"], (
            f'root heuristic_value mismatch!'
            f'Got:      {computed_root["heuristic_value"]}\n'
            f'Expected: {expected_root["heuristic_value"]}'
        )

    with subtests.test('Root node heuristic_run_count'):
        try:
            expected_root_heur_call_count = expected_root["heuristic_call_count"]
        except KeyError:
            expected_root_heur_call_count = expected_root["heuristic_run_count"]
        if expected_root_heur_call_count != computed_root["heuristic_call_count"]:
            pytest.xfail(
                f'root heuristic_run_count mismatch!'
                f'Got:      {computed_root["heuristic_call_count"]}\n'
                f'Expected: {expected_root_heur_call_count}'
            )

    with subtests.test('Root node solution'):
        if expected_root["root_solution"] != computed_root["root_solution"]:
            pytest.xfail(
                f'root root_solution mismatch!'
                f'Got:      {computed_root["root_solution"]}\n'
                f'Expected: {expected_root["root_solution"]}'
            )

    if 'heuristic_data' in expected_root:
        with subtests.test('Root node heuristic data collection'):
            if len(computed_root["heuristic_data"]) != len(expected_root["heuristic_data"]):
                pytest.xfail(
                    f'root len(heuristic data) mismatch!'
                    f'Got:      {len(computed_root["heuristic_data"])}\n'
                    f'Expected: {len(expected_root["heuristic_data"])}'
                )
                
    if 'custom_solver_tests' in expected_result['meta_data']:
        exp_cs_tests = expected_result['meta_data']['custom_solver_tests']
        comp_cs_tests = result['meta_data']['custom_solver_tests']

        for key in exp_cs_tests:
            with subtests.test(key):
                if exp_cs_tests[key] != comp_cs_tests[key]:
                    if exp_cs_tests['sdp_calls'] == 0 and key == 'heuristic_calls':
                        # 4 test cases include the default SDPBound and custom heuristic
                        # in these cases the heuristic calls are called a indeterminant amount
                        # based on the which worker get's which problem with what RNG
                        pytest.xfail(f'(exp - comp) {exp_cs_tests[key]}-{comp_cs_tests[key]}')
                    else:
                        raise ValueError(f'(exp - comp) {exp_cs_tests[key]}-{comp_cs_tests[key]}')
