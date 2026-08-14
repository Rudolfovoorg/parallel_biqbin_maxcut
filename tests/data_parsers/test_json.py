from types import SimpleNamespace

import json
import numpy as np
import pytest

import biqbin.data_parsers as data_parsers
from biqbin import (
    MaxCutFromEdgeWeights,
    MaxCutFromJson,
    MaxCutFromMatrixMarket,
    QuboFromEdgeWeights,
    QuboFromJson,
    QuboFromMatrixMarket,
    QuboFromQPLIB,
)


@pytest.mark.parametrize('parser, matrix_key', [
    (MaxCutFromJson, 'maxcut'),
    (QuboFromJson, 'qubo'),
])
def test_json_reader_rejects_missing_matrix_key(
    tmp_path, parser, matrix_key
):
    path = tmp_path / 'invalid.json'
    path.write_text(json.dumps({}))

    with pytest.raises(KeyError, match=matrix_key):
        parser(str(path)).read()


@pytest.mark.parametrize('parser, matrix_key', [
    (MaxCutFromJson, 'maxcut'),
    (QuboFromJson, 'qubo'),
])
@pytest.mark.parametrize('key', [
    'shape',
    'row',
    'col',
    'data',
])
def test_json_reader_rejects_missing_sparse_keys(
    tmp_path, parser, matrix_key, key
):
    matrix = {
        'shape': [2, 2],
        'row': [0, 1],
        'col': [1, 0],
        'data': [1, 1],
    }

    del matrix[key]

    path = tmp_path / 'invalid.json'
    path.write_text(json.dumps({matrix_key: matrix}))

    with pytest.raises(KeyError, match=key):
        parser(str(path)).read()


@pytest.mark.parametrize('parser, matrix_key', [
    (MaxCutFromJson, 'maxcut'),
    (QuboFromJson, 'qubo'),
])
@pytest.mark.parametrize('row, col', [
    ([-1], [0]),
    ([2], [0]),
    ([0], [-1]),
    ([0], [2]),
])
def test_json_reader_rejects_invalid_sparse_indices(
    tmp_path, parser, matrix_key, row, col
):
    data = {matrix_key: {
            'shape': [2, 2],
            'row': row,
            'col': col,
            'data': [1],
            }}

    path = tmp_path / 'invalid.json'
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match='index'):
        parser(str(path)).read()


@pytest.mark.parametrize('parser, matrix_key', [
    (MaxCutFromJson, 'maxcut'),
    (QuboFromJson, 'qubo'),
])
def test_json_reader_rejects_malformed_sparse_data(
    tmp_path, parser, matrix_key
):
    data = {
        matrix_key: {
            'shape': [2, 2],
            'row': [0, 1],
            'col': [0],
            'data': [1, 2],
        }
    }

    path = tmp_path / 'invalid.json'
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError):
        parser(str(path)).read()


def test_maxcut_json_reader(tmp_path):
    data = {
        'maxcut': {
            'shape': [2, 2],
            'row': [0, 1],
            'col': [1, 0],
            'data': [5, 5],
        }
    }

    path = tmp_path / 'maxcut.json'
    path.write_text(json.dumps(data))

    problem = MaxCutFromJson(str(path)).read()

    np.testing.assert_array_equal(
        problem.maxcut_adjacency_matrix,
        [[0, 5], [5, 0]],
    )


def test_qubo_json_reader(tmp_path):
    data = {
        'qubo': {
            'shape': [2, 2],
            'row': [0, 0, 1],
            'col': [0, 1, 1],
            'data': [1, 2, 3],
        },
        'offset': 4,
        'is_minimization': False,
    }

    path = tmp_path / 'qubo.json'
    path.write_text(json.dumps(data))

    problem = QuboFromJson(str(path)).read()

    np.testing.assert_array_equal(problem.Q, [[1, 2], [0, 3]])
    assert problem.offset == 4
    assert not problem.is_minimization
