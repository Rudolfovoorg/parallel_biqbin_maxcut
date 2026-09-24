import pytest
import numpy as np
from pathlib import Path

from biqbin import (
    MaxCutFromJson,
    MaxCutFromMatrixMarket,
    MaxCutFromEdgeWeights,
    QuboFromJson,
    QuboFromMatrixMarket,
    QuboFromQPLIB,
    ProblemQubo
)
from biqbin.data_parsers import ToFile
from biqbin.utils import qubo_to_qplib_str


MAXCUT_CASES = [
    pytest.param(
        "tests/rudy/g05_60.0.json",
        MaxCutFromMatrixMarket,
        "tests/matrixmarket/g05_60.0.mtx",
        id=f'{MaxCutFromMatrixMarket.__name__}-g05_60.0.mtx'
    ),
    pytest.param(
        "tests/rudy/g05_80.0.json",
        MaxCutFromMatrixMarket,
        "tests/matrixmarket/g05_80.0.mtx",
        id=f'{MaxCutFromMatrixMarket.__name__}-g05_80.0.mtx'
    ),
    pytest.param(
        "tests/rudy/g05_100.4.json",
        MaxCutFromMatrixMarket,
        "tests/matrixmarket/g05_100.4.mtx",
        id=f'{MaxCutFromMatrixMarket.__name__}-g05_100.4.mtx'
    ),
    pytest.param(
        "tests/rudy/g05_60.0.json",
        MaxCutFromEdgeWeights,
        "tests/rudy/g05_60.0",
        id=f'{MaxCutFromEdgeWeights.__name__}-g05_60.0'
    ),
    pytest.param(
        "tests/rudy/g05_80.0.json",
        MaxCutFromEdgeWeights,
        "tests/rudy/g05_80.0",
        id=f'{MaxCutFromEdgeWeights.__name__}-g05_80.0'
    ),
    pytest.param(
        "tests/rudy/g05_100.4.json",
        MaxCutFromEdgeWeights,
        "tests/rudy/g05_100.4",
        id=f'{MaxCutFromEdgeWeights.__name__}-g05_100.4'
    ),
]

QUBO_CASES = [
    pytest.param(
        "tests/qubos/40/kcluster40_025_10_1.json",
        QuboFromMatrixMarket,
        "tests/matrixmarket/kcluster40_025_10_1.mtx",
        id=f'{QuboFromMatrixMarket.__name__}-kcluster40_025_10_1.mtx'
    ),
    pytest.param(
        "tests/qubos/80/kcluster80_025_20_1.json",
        QuboFromMatrixMarket,
        "tests/matrixmarket/kcluster80_025_20_1.mtx",
        id=f'{QuboFromMatrixMarket.__name__}-kcluster80_025_20_1.mtx'
    ),
    pytest.param(
        "tests/qubos/40/kcluster40_025_10_1.json",
        QuboFromQPLIB,
        "tests/qplib/kcluster40_025_10_1.qplib",
        id=f'{QuboFromQPLIB.__name__}-kcluster40_025_10_1.qplib'
    ),
    pytest.param(
        "tests/qubos/80/kcluster80_025_20_1.json",
        QuboFromQPLIB,
        "tests/qplib/kcluster80_025_20_1.qplib",
        id=f'{QuboFromQPLIB.__name__}-kcluster80_025_20_1.qplib'
    ),
]

JSON_TO_QPLIB_WRITE_CASES = [
    "tests/qubos/40/kcluster40_025_10_1.json",
    "tests/qubos/80/kcluster80_025_20_1.json"
]


def _assert_same_matrix(A: np.ndarray, B: np.ndarray, parser_name):
    assert A.shape == B.shape, f"{parser_name} - shape mismatch: {A.shape} vs {B.shape}"

    if not np.array_equal(A, B):
        raise AssertionError(
            f'{parser_name} - '
            f'matrices differ;\n{A=}\n{B=}'
        )


def _test_parsers_equivalence(parser, json_path, other_parser, other_path):
    base_problem = parser(json_path).read()
    other_problem = other_parser(other_path).read()

    A = base_problem.maxcut_adjacency_matrix
    B = other_problem.maxcut_adjacency_matrix

    _assert_same_matrix(A, B, parser_name=f"{other_parser.__name__}")


@pytest.mark.parametrize('json_path, other_parser, other_path', MAXCUT_CASES)
def test_parsers_maxcut_equivalence(json_path, other_parser, other_path):
    _test_parsers_equivalence(MaxCutFromJson,
                              json_path,
                              other_parser,
                              other_path)


@pytest.mark.parametrize("json_path, other_parser, other_path", QUBO_CASES)
def test_parsers_qubo_equivalence(json_path, other_parser, other_path):
    _test_parsers_equivalence(QuboFromJson,
                              json_path,
                              other_parser,
                              other_path)


def _subtest_qplib_writer(save_path: str,
                          qubo: np.ndarray,
                          offset: float,
                          is_minimization: bool,
                          base_problem: ProblemQubo):
    """ qplib problem should have the same maxcut_adjacency_matrix, offset and sense as the base_problem

    Args:
        save_path (str): temporary save path
        qubo (np.ndarray): qubo we are transforming
        offset (float): aka constant term
        is_minimization (bool): objective sense
        base_problem (ProblemQubo): Original problem we are comparing against
    """
    # write
    with open(save_path, 'w') as f:
        f.writelines(qubo_to_qplib_str(qubo,
                                       offset,
                                       is_minimization,
                                       save_path))
    # read
    qplib_problem = QuboFromQPLIB(save_path).read()

    # test
    assert base_problem.offset == qplib_problem.offset
    assert base_problem.is_minimization == qplib_problem.is_minimization
    _assert_same_matrix(base_problem.maxcut_adjacency_matrix,
                        qplib_problem.maxcut_adjacency_matrix,
                        'QPLIB write-read test')


@pytest.mark.parametrize('json_path', JSON_TO_QPLIB_WRITE_CASES)
def test_qplib_writer(json_path, subtests, tmp_path):
    problem = QuboFromJson(json_path).read()
    save_path = str(tmp_path / (Path(json_path).stem + ".qplib"))

    with subtests.test('Upper triangular'):
        _subtest_qplib_writer(save_path, problem.Q,
                              problem.offset, problem.is_minimization, problem)

    with subtests.test('Lower triangular'):
        qT = problem.Q.T
        _subtest_qplib_writer(save_path, qT, problem.offset,
                              problem.is_minimization, problem)

    with subtests.test('Symmetric'):
        sym = (problem.Q + problem.Q.T) / 2
        _subtest_qplib_writer(save_path, sym, problem.offset,
                              problem.is_minimization, problem)


def test_get_output_path(subtests):
    file_with_json = 'tests/rudy/g05_60.0.json'
    file_without_json = 'tests/rudy/g05_60.0'

    with subtests.test('with .json with overwrite'):
        out_file = ToFile.get_output_path(
            out_file=file_with_json, overwrite=True)
        assert (out_file.endswith('.json') and out_file == file_with_json)

    with subtests.test('with .json without overwrite'):
        out_file = ToFile.get_output_path(
            out_file=file_with_json, overwrite=False)
        assert (out_file.endswith('.json') and out_file != file_with_json)

    with subtests.test('without .json with overwrite'):
        out_file = ToFile.get_output_path(
            out_file=file_without_json, overwrite=True)
        assert (out_file.endswith('.json') and out_file ==
                file_without_json + '.json')

    with subtests.test('without .json without overwrite'):
        out_file = ToFile.get_output_path(
            out_file=file_without_json, overwrite=False)
        assert (out_file.endswith('.json') and out_file != file_without_json)
