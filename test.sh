#!/bin/bash

# A test script for comparing of solver output with expected output for a given problem instance.
# Usage: ./test biqbin instance expected_output params

set -e

if [ "$#" -ne 4 ]; then
    echo "Illegal number of parameters"
    echo "Usage:"
    echo "./test biqbin instance expected_output params"
    exit 1
fi

# Filter out Wall clock time value.
filter_out_wall_time_line_value() {
    sed 's/Time = .*/Time = /g'
}

# Filter out Root node bound value.
filter_out_root_node_bound_value() {
    sed 's/Root node bound = .*/Root node bound = /g'
}
# Filter out Solution.
filter_out_bound_again() {
    sed 's/Root node bound:.*/Root node bound: /g'
}

output=$(./biqbin $2 $4) || exit $?
# Filter out value which may vary due to randomnes.
output_filtered=$(echo "$output" | filter_out_wall_time_line_value | filter_out_root_node_bound_value | filter_out_bound_again) || exit $?

expected_output=$(cat $3) || exit $?
# Filter out value which may vary due to randomnes.
expected_output_filtered=$(echo "$expected_output" | filter_out_wall_time_line_value | filter_out_root_node_bound_value | filter_out_bound_again) || exit $?

if [[ "$output_filtered" == "$expected_output_filtered" ]]; then
    echo "O.K."
else
    echo "Failed!"

    diff <(echo "$output_filtered") <(echo "$expected_output_filtered")
    # echo $output_filtered
    # echo "------"
    # echo $expected_output_filtered
    exit 1
fi