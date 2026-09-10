from types import SimpleNamespace
import numpy as np
import pytest

from biqbin_bqp import ProblemBQP, SolutionBQP


def make_bqp(**kwargs):
    data = {
        'F': np.array([[1, 2], [2, 3]], dtype=float),
        'c': np.array([1, 2], dtype=float),
        'A': np.array([[1, 1]], dtype=float),
        'b': np.array([1], dtype=float),
    }
    data.update(kwargs)

    return ProblemBQP(
        **data,
        problem_name='test',
        optimize_input=False,
    )


@pytest.mark.parametrize('kwargs', [{'F': np.array(1)},
                                    {'F': np.array([[1, 2, 3], [2, 3, 4]])},
                                    {'F': np.array([[1, 2], [3, 4]])},
                                    {'c': np.array(1)},
                                    {'c': np.array([[1, 2]])},
                                    {'c': np.array([1, 2, 3])},
                                    {'A': np.array(1)},
                                    {'A': np.array([1, 1])},
                                    {'A': np.array([[1, 1, 1]])},
                                    {'b': np.array(1)},
                                    {'b': np.array([[1]])},
                                    {'b': np.array([1, 2])},],
                         )
def test_bqp_invalid_shapes(kwargs):
    with pytest.raises(ValueError, match='Invalid BQP problem'):
        make_bqp(**kwargs)


@pytest.mark.parametrize('kwargs', [
    {'F': np.array([[1, 0.5], [0.5, 1]])},
    {'c': np.array([1, 1.5])},
    {'A': np.array([[1, 0.5]])},
    {'b': np.array([1.5])},
])
def test_bqp_rejects_non_integer_values(kwargs):
    with pytest.raises(ValueError, match='Invalid BQP problem'):
        make_bqp(**kwargs)


@pytest.mark.parametrize('kwargs', [
    {'F': np.array([[1, np.inf], [np.inf, 1]])},
    {'c': np.array([1, np.nan])},
    {'A': np.array([[1, np.inf]])},
    {'b': np.array([np.nan])},
])
def test_bqp_rejects_non_finite_values(kwargs):
    with pytest.raises(ValueError, match='Invalid BQP problem'):
        make_bqp(**kwargs)


def test_bqp_inputs_are_converted_to_int(monkeypatch):
    import biqbin_bqp

    monkeypatch.setattr(
        biqbin_bqp,
        'interior_point_method_maxcut',
        lambda C: (0.0, None),
    )

    problem = ProblemBQP(
        F=np.zeros((2, 2), dtype=float),
        c=np.zeros(2, dtype=float),
        A=np.array([[1.0, 1.0]]),
        b=np.array([1.0]),
        problem_name='test',
        optimize_input=False,
    )

    assert problem.F.dtype == np.int64
    assert problem.c.dtype == np.int64
    assert problem.A.dtype == np.int64
    assert problem.b.dtype == np.int64


def test_bqp_solution_uses_constraints_to_choose_orientation():
    """ mc_x       = [0, 0]    objective = 0, constraint FAILS
        1 - mc_x   = [1, 1]    objective = 0, constraint PASSES
    """
    problem = SimpleNamespace(
        F=np.zeros((2, 2), dtype=np.int64),
        c=np.zeros(2, dtype=np.int64),
        A=np.array([[1, 0]], dtype=np.int64),
        b=np.array([1], dtype=np.int64),
        rho=1.0,
        const_value=0,
        penalty=0,
        offset=0
    )

    result = {
        'maxcut': {
            'computed_val': 0,
            'x': [0, 0, 0],
        }
    }

    solution = SolutionBQP.__new__(SolutionBQP)
    bqp = solution.get_bqp_solution(result, problem)  # type:ignore

    assert bqp['feasible_solution']
    assert bqp['computed_val'] == 0
    assert bqp['x'] == [1, 1]


def test_bqp_solution_rejects_when_neither_orientation_is_feasible():
    """[0, 0] -> sum = 0
       [1, 1] -> sum = 2
       Neither satisfies x0 + x1 = 1 
    """
    problem = SimpleNamespace(
        F=np.zeros((2, 2), dtype=np.int64),
        c=np.zeros(2, dtype=np.int64),
        A=np.array([[1, 1]], dtype=np.int64),
        b=np.array([1], dtype=np.int64),
        rho=1.0,
        const_value=0,
        penalty=0,
        offset=0
    )

    result = {
        'maxcut': {
            'computed_val': 0,
            'x': [0, 0, 0],
        }
    }

    solution = SolutionBQP.__new__(SolutionBQP)

    with pytest.raises(
        ValueError,
        match='could not be mapped back',
    ):
        solution.get_bqp_solution(result, problem)  # type:ignore


def test_bqp_solution_reports_penalty_infeasibility():
    problem = SimpleNamespace(
        F=np.zeros((2, 2), dtype=np.int64),
        c=np.zeros(2, dtype=np.int64),
        A=np.array([[1, 1]], dtype=np.int64),
        b=np.array([1], dtype=np.int64),
        rho=1.0,
        const_value=10,
        penalty=0,
        offset=0
    )

    result = {
        'maxcut': {
            'computed_val': 0,
            'x': [0, 0, 0],
        }
    }

    solution = SolutionBQP.__new__(SolutionBQP)
    bqp = solution.get_bqp_solution(result, problem)  # type:ignore

    assert not bqp['feasible_solution']
    assert bqp['computed_val'] is None
    assert bqp['x'] is None

