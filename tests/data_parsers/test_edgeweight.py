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


@pytest.mark.parametrize('parser', [
    MaxCutFromEdgeWeights,
    QuboFromEdgeWeights,
])
@pytest.mark.parametrize('content', [
    '2 1\n0 1 4\n',
    '2 1\n3 1 4\n',
    '2 1\n1 0 4\n',
    '2 1\n1 3 4\n',
    '2 1\n1.3 2 4\n',
    '2 1\n1 2.1 4\n',
    '2 1\ninf 2 4\n',
    '2 1\nnan 2 4\n',
    '2 1\n1 inf 4\n',
    '2 1\n1 nan 4\n',
],
    ids=['0-index-row',
         'n+1_row',
         '0-index-col',
         'n+1_col',
         'non-int-col',
         'non-int-row',
         'nan-row',
         'nan-col',
         'inf-row',
         'inf-col'])
def test_edge_reader_rejects_invalid_indices(
    tmp_path, parser, content
):
    path = tmp_path / 'bad.data'
    path.write_text(content)

    with pytest.raises(IndexError):
        parser(str(path)).read()


@pytest.mark.parametrize('parser', [
    MaxCutFromEdgeWeights,
    QuboFromEdgeWeights,
])
@pytest.mark.parametrize('content', [
    '2 1\n1 2 4.3\n',
    '2 1\n1 2 nan\n',
    '2 1\n1 2 inf\n',
])
def test_edge_reader_rejects_invalid_weights(
    tmp_path, parser, content
):
    path = tmp_path / 'bad.data'
    path.write_text(content)

    with pytest.raises(ValueError):
        parser(str(path)).read()


def test_maxcut_edge_reader_single_edge(tmp_path):
    path = tmp_path / 'maxcut.data'
    path.write_text(
        '2 1\n'
        '1 2 4\n'
    )

    problem = MaxCutFromEdgeWeights(str(path)).read()

    np.testing.assert_array_equal(
        problem.maxcut_adjacency_matrix,
        [[0, 4], [4, 0]],
    )


def test_qubo_edge_reader_single_edge(tmp_path):
    path = tmp_path / 'qubo.data'
    path.write_text(
        '2 1\n'
        '1 2 4\n'
    )

    problem = QuboFromEdgeWeights(str(path)).read()

    np.testing.assert_array_equal(
        problem.Q,
        [[0, 4], [0, 0]],
    )
