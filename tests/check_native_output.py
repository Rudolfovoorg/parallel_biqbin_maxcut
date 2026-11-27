"""Check the native solver's 1-based partition against a Gset input graph."""

import re
import sys
from pathlib import Path


def validate(output, graph):
    value = int(re.search(r"^Maximum value = (-?\d+)$", output, re.MULTILINE)[1])
    partition = re.search(r"^Solution = \((.*)\)$", output, re.MULTILINE)[1]
    vertices = [int(v) for v in partition.split()]
    lines = graph.splitlines()
    size, edge_count = map(int, lines[0].split())
    assert len(vertices) == len(set(vertices)), "Duplicate partition vertex"
    assert all(1 <= v <= size for v in vertices), "Partition vertex out of range"
    selected = set(vertices)
    edges = [line.split() for line in lines[1:] if line.strip()]
    assert len(edges) == edge_count, "Incorrect graph edge count"
    cut = sum(int(weight) for u, v, weight in edges if (int(u) in selected) != (int(v) in selected))
    assert cut == value, f"Partition cut value {cut} differs from reported value {value}"


if __name__ == "__main__":
    validate(sys.stdin.read(), Path(sys.argv[1]).read_text())
