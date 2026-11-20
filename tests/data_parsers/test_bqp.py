import json

import numpy as np
import pytest

import biqbin.data_parsers as data_parsers


@pytest.mark.parametrize(
    "content, match",
    [
        ("2 1\nF\n0 1 5\n", "index 0 out of range"),
        ("2 1\nF\n3 1 5\n", "index 3 out of range"),
        ("2 1\nc\n3 5\n", "index 3 out of range"),
        ("2 1\nA\n2 1 5\n", "index 2 out of range"),
        ("2 1\nb\n2 5\n", "index 2 out of range"),
    ],
)
def test_bqp_file_reader_rejects_invalid_indices(tmp_path, content, match):
    path = tmp_path / "bad.bqp"
    path.write_text(content)

    with pytest.raises(IndexError, match=match):
        data_parsers.BQPFromBQPFile(str(path)).read()


@pytest.mark.parametrize(
    "content, match",
    [
        ("", "is empty"),
        ("foo bar\n", "first line must be"),
        ("2 1\n1 1 5\n", "data before any section header"),
        ("2 1\nF\n1 x 5\n", "section"),
        ("2 1\nF\n1 1\n", "section"),
    ],
)
def test_bqp_file_reader_rejects_invalid_values(tmp_path, content, match):
    path = tmp_path / "bad.bqp"
    path.write_text(content)

    with pytest.raises(ValueError, match=match):
        data_parsers.BQPFromBQPFile(str(path)).read()


def test_bqp_file_reader(tmp_path, monkeypatch):
    import biqbin.problem as problem_module

    monkeypatch.setattr(
        problem_module,
        "interior_point_method_maxcut",
        lambda C: (0.0, None),
    )

    path = tmp_path / "test.bqp"
    path.write_text("2 1\nF\n1 1 1\n1 2 2\n2 2 3\nc\n1 4\n2 5\nA\n1 1 1\n1 2 1\nb\n1 1\n")

    problem = data_parsers.BQPFromBQPFile(str(path)).read()

    np.testing.assert_array_equal(problem.F, [[1, 2], [2, 3]])
    np.testing.assert_array_equal(problem.c, [4, 5])
    np.testing.assert_array_equal(problem.A, [[1, 1]])
    np.testing.assert_array_equal(problem.b, [1])


@pytest.mark.parametrize(
    "key, entries",
    [
        ("F", [[-1, 0, 1]]),
        ("F", [[2, 0, 1]]),
        ("c", [[-1, 1]]),
        ("c", [[2, 1]]),
        ("A", [[-1, 0, 1]]),
        ("A", [[1, 0, 1]]),
        ("A", [[0, -1, 1]]),
        ("A", [[0, 2, 1]]),
        ("b", [[-1, 1]]),
        ("b", [[1, 1]]),
    ],
)
def test_bqp_json_reader_rejects_invalid_indices(tmp_path, key, entries):

    data = {
        "number_of_variables": 2,
        "number_of_constraints": 1,
        "F": [],
        "c": [],
        "A": [],
        "b": [],
    }
    data[key] = entries

    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data))

    with pytest.raises(IndexError, match="out of range"):
        data_parsers.BQPFromJson(str(path)).read()


@pytest.mark.parametrize(
    "key",
    [
        "number_of_variables",
        "number_of_constraints",
        "F",
        "c",
        "A",
        "b",
    ],
)
def test_bqp_json_reader_rejects_missing_keys(tmp_path, key):
    data = {
        "number_of_variables": 2,
        "number_of_constraints": 1,
        "F": [],
        "c": [],
        "A": [],
        "b": [],
    }
    del data[key]

    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data))

    with pytest.raises(KeyError, match=key):
        data_parsers.BQPFromJson(str(path)).read()


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize(
    "reader,filename,offset",
    [
        (data_parsers.BQPFromBQPFile, "tests/bqp/test_bqp.data", 0),
        (data_parsers.BQPFromJson, "tests/bqp/test_bqp.json", 0),
        (data_parsers.BQPFromQBODense, "tests/bqp/kcluster40_025_10_1_QBO_dense.json", 0),
        (data_parsers.BQPFromQBOSparse, "tests/bqp/kcluster40_025_10_1_QBO_sparse.json", 7),
    ],
)
def test_bqp_reader_preserves_offset_and_optimization(tmp_path, reader, filename, offset, optimize):
    # Keep the real SDP-based transformation: compare the same input before and
    # after GCD scaling, including a nonzero offset in the format supporting it.
    if offset:
        from pathlib import Path

        data = json.loads(Path(filename).read_text())
        data["QBO"]["offset"] = offset
        path = tmp_path / "offset.json"
        path.write_text(json.dumps(data))
        filename = str(path)
    reference = reader(filename, optimize_input=False).read()
    problem = reader(filename, optimize_input=optimize).read()
    assert problem.offset == reference.offset == offset
    assert problem.optimize_mc_adj_matrix is optimize
    np.testing.assert_array_equal(
        problem.maxcut_adjacency_matrix * problem.gcd, reference.maxcut_adjacency_matrix
    )
    np.testing.assert_array_equal(problem.F, reference.F)
    np.testing.assert_array_equal(problem.c, reference.c)
