#!/bin/bash

# A test script for comparing solver output with expected output for a given problem instance.
# Usage: ./test.sh biqbin instance expected_output params

set -e

if [ "$#" -ne 4 ]; then
    echo "Illegal number of parameters"
    echo "Usage:"
    echo "./test.sh biqbin instance expected_output params"
    exit 1
fi

# Extract only the important lines for strict comparison
extract_comparison_lines() {
    grep -E '^(Root node bound =|Maximum value =|Solution =)'
}

# Extract informational lines only
extract_info_lines() {
    grep -E '^(Nodes =|Time =)'
}

# Run solver and capture output
output=$(mpiexec -n 8 ./$1 "$2" "$4") || exit $?

# Extract for comparison
output_filtered=$(echo "$output" | extract_comparison_lines)

# Extract info
nodes=$(echo "$output" | grep '^Nodes =' | sed 's/Nodes = //')
time_taken=$(echo "$output" | grep '^Time =' | sed 's/Time = //' | sed 's/s$//')

# Ensure variables aren't empty (default to 0)
nodes=${nodes:-0}
time_taken=${time_taken:-0}

expected_nodes=$(cat "$3" | grep '^Nodes =' | sed 's/Nodes = //')
expected_time=$(cat "$3" | grep '^Time =' | sed 's/Time = //' | sed 's/s$//')

# Ensure expected values aren't empty (default to 0)
expected_nodes=${expected_nodes:-0}
expected_time=${expected_time:-0}

# Calculate differences
node_diff=$((nodes - expected_nodes))
time_diff=$(awk "BEGIN {print $time_taken - $expected_time}")

# Filter output (you had this missing)
output_filtered=$(echo "$output" | extract_comparison_lines)
expected_output_filtered=$(cat "$3" | extract_comparison_lines)

# Print result
if [[ "$output_filtered" == "$expected_output_filtered" ]]; then
    echo "O.K.! Nodes diff = ${node_diff}; Time diff = ${time_diff}s"
else
    echo "Failed!"
    diff <(echo "$output_filtered") <(echo "$expected_output_filtered")
    exit 1
fi
