"""Validate reported solutions against input matrices, independently of reconstruction."""

from pathlib import Path

import numpy as np
import pytest

from biqbin.data_parsers import MaxCutFromJson, QuboFromJson, QuboFromQPLIB
from biqbin_bqp import BQPFromBQPFile, BQPFromJson, BQPFromQBODense, BQPFromQBOSparse


def load_problem(result):
    filename = result["meta_data"]["instance"]
    if "bqp" in result:
        if filename.endswith("_dense.json"):
            reader = BQPFromQBODense
        elif filename.endswith("_sparse.json"):
            reader = BQPFromQBOSparse
        else:
            reader = BQPFromJson if Path(filename).suffix == ".json" else BQPFromBQPFile
    elif "qubo" in result:
        reader = QuboFromQPLIB if Path(filename).suffix == ".qplib" else QuboFromJson
    else:
        reader = MaxCutFromJson
    return reader(filename).read()


def assert_binary_vector(values, size):
    x = np.asarray(values)
    assert x.shape == (size,), f"Expected {size} binary values, got shape {x.shape}"
    assert np.all((x == 0) | (x == 1)), "Solution must contain only 0 and 1"
    return x


def assert_cut(result, adjacency):
    x = assert_binary_vector(result["x"], adjacency.shape[0])
    # Sum each crossing edge once, independently of the solver's objective formula.
    cut = np.triu(adjacency, 1)[x[:, None] != x[None, :]].sum()
    assert result["computed_val"] == pytest.approx(cut, rel=0, abs=1e-6)
    if "solution" in result:
        assert result["solution"] == (np.flatnonzero(x) + 1).tolist()
