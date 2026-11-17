#!/bin/bash
# Usage: test.sh "launcher executable" instance expected_output params
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo 'Usage: test.sh "launcher executable" instance expected_output params' >&2
    exit 2
fi

# Split the launcher argument into words without evaluating shell code.
read -r -a command <<< "$1"
if output=$("${command[@]}" "$2" "$4" 2>&1); then
    printf '%s\n' "$output"
else
    status=$?
    printf '%s\n' "$output" >&2
    exit "$status"
fi

# Missing results must fail, even if the expected file is also malformed.
for pattern in '^Maximum value = -?[0-9]+$' '^Solution = \(.*\)$'; do
    if [[ $(printf '%s\n' "$output" | grep -Ec "$pattern" || true) != 1 ]] ||
       [[ $(grep -Ec "$pattern" "$3" || true) != 1 ]]; then
        echo "Missing, malformed, or duplicate result for $2 ($pattern)" >&2
        exit 1
    fi
done

# Preserve the existing strict regression comparison, ignoring timing and nodes.
diff -u <(grep -E '^(Maximum value =|Solution =)' "$3") \
        <(printf '%s\n' "$output" | grep -E '^(Maximum value =|Solution =)')
