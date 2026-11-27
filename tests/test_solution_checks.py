"""Alternative optima pass; malformed or incorrectly scored cuts fail."""

import numpy as np
import pytest

from tests.solution_checks import assert_cut


@pytest.mark.parametrize("x", [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]])
def test_alternative_optimal_cuts(x):
    adjacency = np.ones((3, 3)) - np.eye(3)
    assert_cut({"x": x, "computed_val": 2}, adjacency)


@pytest.mark.parametrize(
    "result",
    [
        {"x": [0, 0, 0], "computed_val": 2},
        {"x": [2, 0, 0], "computed_val": 2},
        {"x": [1, 0], "computed_val": 2},
        {"x": [float("nan"), 0, 0], "computed_val": 2},
        {"x": [1, 0, 0], "computed_val": 2, "solution": [2]},
    ],
)
def test_invalid_cuts_fail(result):
    with pytest.raises(AssertionError):
        assert_cut(result, np.ones((3, 3)) - np.eye(3))
