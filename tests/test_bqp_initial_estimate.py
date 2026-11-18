"""Unit tests for BQP initial-estimate conversion (no MPI job required)."""

from types import SimpleNamespace

import numpy as np
import pytest

import biqbin.biqbin_base as solver_module
from biqbin.biqbin_base import BQPSolver


@pytest.fixture
def solver(monkeypatch):
    # Avoid MPI setup and native callback registration: this method needs
    # only its problem instance. Catch aborts without terminating pytest.
    def abort(code):
        raise RuntimeError(f"MPI abort: {code}")

    monkeypatch.setattr(solver_module, "abort_mpi", abort)

    obj = BQPSolver.__new__(BQPSolver)
    problem = SimpleNamespace(
        F=np.zeros((2, 2), dtype=np.int64),
        c=np.zeros(2, dtype=np.int64),
        A=np.array([[1, 0]], dtype=np.int64),
        b=np.array([1], dtype=np.int64),
        # An edge to the auxiliary vertex distinguishes a vector from
        # its complement when the auxiliary vertex is held at zero.
        maxcut_adjacency_matrix=np.array([[0, 0, 2], [0, 0, 0], [2, 0, 0]], dtype=np.int64),
        gcd=1,
        const_value=2,
    )
    obj._BQPSolver__problem = problem
    return obj


def test_bqp_initial_estimate_uses_direct_orientation(solver):
    result = solver._initial_estimate_bqp_to_maxcut(np.array([1, 0]))
    np.testing.assert_array_equal(result, [1, 0, 0])


def test_bqp_initial_estimate_uses_complement_when_objective_requires_it(solver):
    solver.problem.const_value = 0
    result = solver._initial_estimate_bqp_to_maxcut(np.array([1, 0]))
    np.testing.assert_array_equal(result, [0, 1, 0])


@pytest.mark.parametrize(
    "estimate",
    [
        [1],
        [1, 0, 1],
        [[1, 0]],
        [1, 0.5],
        [1, np.nan],
        [1, np.inf],
    ],
)
def test_bqp_initial_estimate_rejects_invalid_binary_vectors(solver, estimate):
    with pytest.raises(RuntimeError, match="MPI abort: 10"):
        solver._initial_estimate_bqp_to_maxcut(estimate)


def test_bqp_initial_estimate_rejects_constraint_violation(solver):
    with pytest.raises(RuntimeError, match="MPI abort: 10"):
        solver._initial_estimate_bqp_to_maxcut([0, 1])


def test_bqp_initial_estimate_rejects_inconsistent_objective(solver):
    solver.problem.const_value = 17  # Neither candidate has cut value 17.
    with pytest.raises(RuntimeError, match="MPI abort: 10"):
        solver._initial_estimate_bqp_to_maxcut([1, 0])
