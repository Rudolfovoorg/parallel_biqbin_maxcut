import sys
import json
import warnings
import pytest
"""
    Compares Biqbin output with expected output for all Python versions
"""


def test_biqbin_output(problem_instance, request):
    """
    Compare Biqbin output with expected output for one instance.
    """

    with open(problem_instance + '.output.json', 'r') as f:
        result = json.load(f)

    with open(problem_instance + '-expected_output.json', 'r') as f:
        expected_result = json.load(f)

    without_sol_vec = request.config.getoption('--without-sol-vector')

    # Meta data
    bab_nodes_diff = expected_result['meta_data']['eval_bab_nodes'] - \
        result['meta_data']['eval_bab_nodes']
    time_diff = expected_result['meta_data']['time'] - \
        result['meta_data']['time']

    print(f'Bab nodes diff = {bab_nodes_diff} Time diff = {time_diff:.3f}')

    # Check if non-branching instances branched
    if expected_result['meta_data']['eval_bab_nodes'] == 1:
        assert bab_nodes_diff == 0, (
            f'Bab nodes mismatch!\n',
            f'Got:      {result['meta_data']['eval_bab_nodes']}\n'
            f'Expected: {expected_result['meta_data']['eval_bab_nodes']}\n'
        )

    # --- Check maxcut ---
    assert expected_result['maxcut']['computed_val'] == result['maxcut']['computed_val'], (
        f'maxcut mismatch!\n'
        f'Got:      {result['maxcut']['computed_val']}\n'
        f'Expected: {expected_result['maxcut']['computed_val']}\n'
        f'Got:      {result['maxcut']['x']}\n'
        f'Expected: {expected_result['maxcut']['x']}'
    )

    if not without_sol_vec:
        assert expected_result['maxcut']['x'] == result['maxcut']['x'], (
            f'maxcut mismatch!\n'
            f'Got:      {result['maxcut']['x']}\n'
            f'Expected: {expected_result['maxcut']['x']}'
        )
        assert expected_result['maxcut']['solution'] == result['maxcut']['solution'], (
            f'maxcut mismatch!\n'
            f'Got:      {result['maxcut']['solution']}\n'
            f'Expected: {expected_result['maxcut']['solution']}'
        )

    # --- Check qubo if present ---
    if 'qubo' in expected_result:
        assert expected_result['qubo']['computed_val'] == result['qubo']['computed_val'], (
            f'qubo mismatch!\n'
            f'Got solution: {result['qubo']['computed_val']}\n'
            f'Expected solution: {expected_result['qubo']['computed_val']}\n'
            f'Got solution: {result['qubo']['x']}\n'
            f'Expected solution: {expected_result['qubo']['x']}'
        )
        if not without_sol_vec:
            assert expected_result['qubo']['x'] == result['qubo']['x'], (
                f'qubo mismatch!\n'
                f'Got solution: {result['qubo']['x']}\n'
                f'Expected solution: {expected_result['qubo']['x']}'
            )
            assert expected_result['qubo']['solution'] == result['qubo']['solution'], (
                f'qubo mismatch!\n'
                f'Got solution: {result['qubo']['solution']}\n'
                f'Expected solution: {expected_result['qubo']['solution']}'
            )

    # --- Check bqp if present ---
    if 'bqp' in expected_result:
        assert expected_result['bqp'] == result['bqp'], (
            f'bqp mismatch!\n'
            f'Got: {result['bqp']}\n'
            f'Expected: {expected_result['bqp']}'
        )

    if 'heuristic_data' in expected_result['meta_data']:
        assert len(result['meta_data']['root_node']['heuristic_data']) == len(expected_result['meta_data']['heuristic_data']), (
            f'root len(heuristic data) mismatch!'
            f'Got:      {len(result['meta_data']['heuristic_data'])}\n'
            f'Expected: {len(expected_result['meta_data']['heuristic_data'])}'
        )
