#!/bin/bash

# A test script for comparing solver output with expected output for a given problem instance.
# Usage: ./test.sh biqbin instance expected_output params

if [ "$#" -ne 4 ]; then
    echo "Illegal number of parameters"
    echo "Usage:"
    echo "./test.sh biqbin instance expected_output params"
    exit 1
fi

# Extract only the important lines for strict comparison
extract_comparison_lines() {
    grep -E '^(Maximum value =|Solution =)'
}

# Extract informational lines only
extract_info_lines() {
    grep -E '^(Nodes =|Time =)'
}

# Run solver and capture output
output=$($1 $2 $4) || exit $?

# Extract for comparison
output_filtered=$(echo "$output" | extract_comparison_lines)

# Extract info
nodes=$(echo "$output" | grep '^Nodes =' | sed 's/Nodes = //')
root_node_bound=$(echo "$output" | grep '^Root node bound =' | sed 's/Root node bound = //')
time_taken=$(echo "$output" | grep '^Time =' | sed 's/Time = //' | sed 's/s$//')
max_val=$(echo "$output" | grep '^Maximum value =' | sed 's/Maximum value = //')
min_val=$(echo "$output" | grep '^Qubo Minimum Value =' | sed 's/Qubo Minimum Value = //')

# Ensure variables aren't empty (default to 0)
nodes=${nodes:-0}
time_taken=${time_taken:-0}

expected_nodes=$(cat "$3" | grep '^Nodes =' | sed 's/Nodes = //')
expected_root_node_bound=$(cat "$3" | grep '^Root node bound =' | sed 's/Root node bound = //')
expected_time=$(cat "$3" | grep '^Time =' | sed 's/Time = //' | sed 's/s$//')

exp_max_val=$(cat "$3" | grep '^Maximum value =' | sed 's/Maximum value = //')


# Ensure expected values aren't empty (default to 0)
expected_nodes=${expected_nodes:-0}
expected_time=${expected_time:-0}

# Calculate differences
node_diff=$((nodes - expected_nodes))
time_diff=$(awk "BEGIN {print $time_taken - $expected_time}")

max_val_diff=$(($max_val - $exp_max_val))
root_node_bound_diff=$(awk "BEGIN {print $root_node_bound - $expected_root_node_bound}")

output_filtered=$(echo "$output" | extract_comparison_lines)
expected_output_filtered=$(cat "$3" | extract_comparison_lines)

# Print result
if [[ "$output_filtered" == "$expected_output_filtered" ]]; then
    echo "O.K - ${2} Max val diff = ${max_val_diff}; Node diff = ${node_diff}; Root bound diff = ${root_node_bound_diff} Time diff = ${time_diff}s" 
else
    echo "Failed! $2"
    echo "Max val diff = ${max_val_diff}; Node diff = ${node_diff}; Root bound diff = ${root_node_bound_diff} Time diff = ${time_diff}s" 
    diff <(echo "$output_filtered") <(echo "$expected_output_filtered")
fi
