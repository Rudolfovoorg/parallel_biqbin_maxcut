import numpy as np
import pytest

from biqbin import ProblemMaxCut, SolutionMaxCut


@pytest.mark.parametrize('matrix', [np.array(1),
                                    np.array([1, 2]),
                                    np.array([[1, 2, 3], [2, 3, 4]]),],
                         ids=['scalar', 'vector', 'non-square'])
def test_maxcut_rejects_invalid_shapes(matrix):
    with pytest.raises(ValueError, match=r'(shape|Dimension)'):
        ProblemMaxCut(matrix, problem_name='test')


@pytest.mark.parametrize('matrix', [
    np.array([[0, 0.5], [0.5, 0]]),
    np.array([[0, np.nan], [np.nan, 0]]),
    np.array([[0, np.inf], [np.inf, 0]]),],
    ids=['non-integer', 'nan', 'inf'])
def test_maxcut_rejects_invalid_values(matrix):
    with pytest.raises(ValueError, match='integer'):
        ProblemMaxCut(matrix, problem_name='test')


def test_maxcut_optimization_divides_by_gcd():
    matrix = np.array([
        [0.0, 6.0],
        [6.0, 0.0],
    ])

    problem = ProblemMaxCut(
        matrix,
        problem_name='test',
        optimize_mc_adj_matrix=True,
    )

    assert problem.gcd == 6
    np.testing.assert_array_equal(
        problem.maxcut_adjacency_matrix,
        [[0, 1], [1, 0]],
    )


def test_maxcut_solution_restores_gcd():
    matrix = np.array([
        [0.0, 6.0],
        [6.0, 0.0],
    ])

    problem = ProblemMaxCut(
        matrix,
        problem_name='test',
        optimize_mc_adj_matrix=True,
    )

    result = {
        'maxcut': {
            'computed_val': 1,
            'solution': [1],
            'x': [1, 0],
        },
        'meta_data': {},
    }

    solution = SolutionMaxCut(result, problem)

    assert solution.solution['computed_val'] == 6
    assert solution.solution['solution'] == [1]
    assert solution.solution['x'] == [1, 0]


def test_maxcut_optimization_accepts_integer_matrix():
    problem = ProblemMaxCut(
        np.array([
            [0, 6],
            [6, 0],
        ], dtype=np.int64),
        problem_name='test',
        optimize_mc_adj_matrix=True,
    )

    assert problem.gcd == 6
