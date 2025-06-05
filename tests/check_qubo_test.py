import sys
import json

"""
    Compares expected output in qubo json file to the computed one
"""
if __name__ == '__main__':
    _, problem_instance_file_name = sys.argv

    with open(problem_instance_file_name + ".output.json",  "r") as f:
        result = json.load(f)

    with open(problem_instance_file_name + "-expected_output.json",  "r") as f:
        expected_result = json.load(f)
    
    test_failed = False
    diffs = {
        "time": expected_result["time"] - result["time"]
    }
    for problem in ["maxcut", "qubo"]:
        com_val = result[problem]["computed_val"]
        exp_val = expected_result[problem]["computed_val"]

        val_diff = com_val - exp_val
        diffs[f"{problem}_val"] = val_diff

        com_x = list(float(i) for i in result[problem]["x"])
        com_len_x = sum(com_x)
        exp_x = expected_result[problem]["x"]
        exp_len_x = sum(exp_x)
        if abs(val_diff) > 0.0001:
            print(f"Computed x = {com_x}")
            print(f"Expected x = {exp_x}")
            test_failed = True
    if not test_failed:
        print(
            f"OK! - {problem_instance_file_name} Qubo val diff = {diffs["qubo_val"]}; nodes diff = {com_len_x - exp_len_x}; Time diff = {diffs["time"]}"
        )
        exit(0)
    else:
        print(
        f"FAILED! - {problem_instance_file_name} Maxcut diff = {diffs["maxcut_val"]}; Qubo diff = {diffs["qubo_val"]};  Time diff = {diffs["time"]}"
        )
        exit(1)