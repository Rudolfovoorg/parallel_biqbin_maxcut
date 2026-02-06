import sys
import json
import warnings

"""
    Compares Biqbin output with expected output for all Python versions
"""

if __name__ == '__main__':
    test_failed = False
    _, problem_instance = sys.argv

    with open(problem_instance + ".output.json",  "r") as f:
        result = json.load(f)

    with open(problem_instance + '-expected_output.json',  "r") as f:
        expected_result = json.load(f)

    # Check maxcut result
    if not expected_result['maxcut'] == result['maxcut']:
        warnings.warn(f'maxcut result != expected result!')
        warnings.warn(f'{result['maxcut']=}\n{expected_result['maxcut']=}')
        test_failed = True

    # If qubo check qubo result
    if "qubo" in expected_result:
        if not expected_result['qubo'] == result['qubo']:
            warnings.warn(f'qubo result != expected result!')
            warnings.warn(f'{result['qubo']=}\n{expected_result['qubo']=}')
            test_failed = True
    
    # if bqp check bqp
    if "bqp" in expected_result:
        if not expected_result['bqp'] == result['bqp']:
            warnings.warn(f'bqp result != expected result!')
            warnings.warn(f'{result['bqp']=}\n{expected_result['bqp']=}')
            test_failed = True

    # Check metadata for result
    bab_nodes_diff = expected_result['meta_data']['eval_bab_nodes'] - \
        result['meta_data']['eval_bab_nodes']
    workers_used_diff = expected_result['meta_data']['num_workers_used'] - \
        result['meta_data']['num_workers_used']

    # if bab_nodes_diff != 0:
    #     warnings.warn(
    #         f'{result['meta_data']['eval_bab_nodes']=} != {expected_result['meta_data']['eval_bab_nodes']=}')
    #     test_failed = True

    # if workers_used_diff != 0:
    #     warnings.warn(
    #         f'{result['meta_data']['num_workers_used']=} != {expected_result['meta_data']['num_workers_used']=}')
    #     test_failed = True

    time_diff = expected_result['meta_data']['time'] - \
        result['meta_data']['time']

    if not test_failed:
        print(
            f'OK! - {problem_instance}; Bab nodes diff = {bab_nodes_diff}; Time diff = {time_diff}'
        )
        exit(0)
    else:
        print(
            f'FAILED! - {problem_instance}; Bab nodes diff = {bab_nodes_diff}; Time diff = {time_diff}'
        )
        exit(1)
