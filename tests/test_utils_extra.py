"""Focused tests for previously uncovered utility paths."""

import numpy as np
import pytest

from biqbin.utils import (
    check_matrix_validity,
    convert_numpy_to_json_serializable,
    data_collector,
    from_sparse,
    to_sparse,
)


@pytest.mark.parametrize(
    "matrix",
    [
        np.zeros((2, 2), dtype=int),
        np.diag([3, -2]).astype(float),
        np.array([[0, -3], [5, 0]], dtype=int),
    ],
)
def test_sparse_round_trip(matrix):
    representation = to_sparse(matrix)
    assert representation["shape"] == matrix.shape
    np.testing.assert_array_equal(from_sparse(representation), matrix)


@pytest.mark.parametrize(
    ("matrix", "error"),
    [
        ([[1, 2], [3, 4]], TypeError),
        (np.array([1, 2]), ValueError),
        (np.ones((2, 3)), ValueError),
        (np.array([[1, 0.5], [0, 1]]), ValueError),
        (np.array([[1, np.nan], [0, 1]]), ValueError),
        (np.array([[1, np.inf], [0, 1]]), ValueError),
        (np.array([[True, False], [False, True]]), TypeError),
    ],
)
def test_matrix_validation_rejects_bad_input(matrix, error):
    with pytest.raises(error):
        check_matrix_validity(matrix)


def test_data_collector_disabled_and_enabled():
    class Worker:
        enabled = False

        @data_collector("enabled", "records")
        def compute(self, value):
            return value + 1

    worker = Worker()
    assert worker.compute(2) == 3
    assert not hasattr(worker, "records")

    worker.enabled = True
    assert worker.compute(4) == 5
    assert len(worker.records) == 1
    assert worker.records[0]["value"] == 5
    assert worker.records[0]["time"] >= 0


def test_data_collector_does_not_record_failed_calls():
    class Worker:
        enabled = True

        @data_collector("enabled", "records")
        def compute(self):
            raise ValueError("failed")

    worker = Worker()
    with pytest.raises(ValueError, match="failed"):
        worker.compute()
    assert worker.records == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (np.array([1, 2]), [1, 2]),
        (np.int64(7), 7),
        (np.float64(1.5), 1.5),
    ],
)
def test_json_conversion(value, expected):
    assert convert_numpy_to_json_serializable(value) == expected


def test_json_conversion_rejects_unknown_type():
    with pytest.raises(TypeError, match="not JSON serializable"):
        convert_numpy_to_json_serializable(object())
