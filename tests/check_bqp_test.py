import sys
import json

"""
    Compares expected output in bqp json file to the computed one
"""
if __name__ == '__main__':
    _, problem_instance_file_name = sys.argv

    with open(problem_instance_file_name + ".output.json",  "r") as f:
        result = json.load(f)

    with open(problem_instance_file_name + "-expected_output.json",  "r") as f:
        expected_result = json.load(f)
    
    test_failed = False
    diffs = {
        "time": expected_result["meta_data"]["time"] - result["meta_data"]["time"]
    }
    for problem in ["maxcut", "bqp"]:
        com_val = result[problem]["computed_val"]
        exp_val = expected_result[problem]["computed_val"]

        val_diff = com_val - exp_val
        diffs[f"{problem}_val"] = val_diff

        com_x = list(float(i) for i in result[problem]["x"])
        com_len_x = sum(com_x)
        exp_x = expected_result[problem]["x"]
        exp_len_x = sum(exp_x)
        diffs['len_diffs'] = abs(com_len_x - exp_len_x)
        if abs(val_diff) > 0.0001:
            print(f"Computed x = {com_x}")
            print(f"Expected x = {exp_x}")
            test_failed = True
        test_failed = not all(result[problem]["x"][i] == expected_result[problem]["x"][i] for i in range(len(result[problem]["x"])))
    
        if not result["bqp"]["feasible_solution"] == expected_result["bqp"]["feasible_solution"]:
            test_failed = True
        
        if not result["bqp"]["rho"] == expected_result["bqp"]["rho"]:
            test_failed = True
            print(f"Expected rho {expected_result["bqp"]["rho"]} != computed rho {result["bqp"]["rho"]}")
        
        if not result["bqp"]["rho"] == expected_result["bqp"]["rho"]:
            test_failed = True
            print(f"Expected rho {expected_result["bqp"]["rho"]} != computed rho {result["bqp"]["rho"]}")
        
        if not result["bqp"]["const_value"] == expected_result["bqp"]["const_value"]:
            test_failed = True
            print(f"Expected const_value {expected_result["bqp"]["const_value"]} != computed const_value {result["bqp"]["const_value"]}")
        
    if not test_failed:
        print(
            f"OK! - {problem_instance_file_name} bqp val diff = {diffs["bqp_val"]}; bqp diff = {diffs['len_diffs']}; Time diff = {diffs["time"]}"
        )
        exit(0)
    else:
        print(
        f"FAILED! - {problem_instance_file_name} Maxcut diff = {diffs["maxcut_val"]}; bqp diff = {diffs["len_diffs"]};  Time diff = {diffs["time"]}"
        )
        exit(1)