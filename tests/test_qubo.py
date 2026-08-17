import numpy as np
import pytest

from biqbin import ProblemQubo, SolutionQubo


def make_qubo(**kwargs):
    data = {
        'Q': np.array([[1, 2], [0, 3]], dtype=float),
        'offset': 0,
        'problem_name': 'test',
        'is_minimization': True,
    }
    data.update(kwargs)

    return ProblemQubo(**data)


def test_qubo_to_maxcut_conversion():
    Q = np.array([
        [1, 2],
        [0, 3],
    ])

    problem = ProblemQubo(
        Q=Q,
        offset=4,
        problem_name='test',
        is_minimization=True,
    )

    np.testing.assert_array_equal(
        problem.maxcut_adjacency_matrix,
        [
            [0, 1, -2],
            [1, 0, -4],
            [-2, -4, 0],
        ],
    )


def test_qubo_maximization_negates_maxcut_problem():
    Q = np.array([
        [1, 2],
        [0, 3],
    ])

    min_problem = ProblemQubo(
        Q=Q,
        offset=0,
        problem_name='min',
        is_minimization=True,
    )
    max_problem = ProblemQubo(
        Q=Q,
        offset=0,
        problem_name='max',
        is_minimization=False,
    )

    np.testing.assert_array_equal(
        max_problem.maxcut_adjacency_matrix,
        -min_problem.maxcut_adjacency_matrix,
    )


@pytest.mark.parametrize('maxcut_solution', [
    [2, 3],
    [1],
])
def test_qubo_solution_conversion(maxcut_solution):
    problem = ProblemQubo(
        Q=np.array([
            [1, 2],
            [0, 3],
        ]),
        offset=4,
        problem_name='test',
        is_minimization=True,
    )

    result = {
        'maxcut': {
            'computed_val': 0,
            'solution': maxcut_solution,
            'x': [],
        },
        'meta_data': {},
    }

    solution = SolutionQubo(result, problem)

    assert solution.solution['x'] == [1, 0]
    assert solution.solution['solution'] == [1]
    assert solution.solution['computed_val'] == 5
    assert solution.solution['offset'] == 4
    assert solution.solution['cardinality'] == 1
    assert solution.solution['minimization']


@pytest.mark.parametrize('Q', [np.array(1),
                               np.array([1, 2]),
                               np.array([[1, 2, 3], [2, 3, 4]]),],
                         ids=['scalar', 'vector', 'non-square'])
def test_qubo_rejects_invalid_shapes(Q):
    with pytest.raises(ValueError, match=r"(shape|Dimension)"):
        make_qubo(Q=Q)


@pytest.mark.parametrize('Q', [
    np.array([[1, 0.5], [0, 1]]),
    np.array([[1, np.nan], [0, 1]]),
    np.array([[1, np.inf], [0, 1]]),],
    ids=['non-integer', 'nan', 'inf'])
def test_qubo_rejects_invalid_values(Q):
    with pytest.raises(ValueError, match="integer"):
        make_qubo(Q=Q)
