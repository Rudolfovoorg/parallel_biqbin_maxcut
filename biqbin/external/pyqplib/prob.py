import numpy as np
import scipy as sp

from .cons import Constraints
from .desc import ProblemDescription
from .init_vals import InitialValues
from .obj import Objective
from .util import sparse_zero


class Problem:
    def __init__(
        self,
        problem_desc,
        initial_values,
        objective,
        constraints,
        var_lb,
        var_ub,
        var_types,
    ):
        self.var_lb = var_lb
        self.var_ub = var_ub
        self._desc = problem_desc

        if constraints is None:
            self.num_cons = 0
        else:
            self.num_cons = constraints.lb.shape[0]

        (self.num_vars,) = var_lb.shape

        assert var_ub.shape == var_lb.shape

        self.var_types = var_types

        assert initial_values.num_cons == self.num_cons
        assert initial_values.num_vars == self.num_vars

        self._initial_values = initial_values
        self._objective = objective
        self._constraints = constraints

    @property
    def name(self) -> str:
        return self.desc.name

    @property
    def description(self) -> ProblemDescription:
        return self._desc

    @property
    def initial_values(self) -> InitialValues:
        return self._initial_values

    @property
    def cons_lb(self) -> np.ndarray:
        if self._constraints is None:
            return np.array([])
        return self._constraints.lb

    @property
    def x0(self) -> np.ndarray:
        return self.initial_values.x0

    @property
    def y0(self) -> np.ndarray:
        return self.initial_values.y0

    @property
    def z0(self) -> np.ndarray:
        return self.initial_values.z0

    @property
    def cons_ub(self) -> np.ndarray:
        if self._constraints is None:
            return np.array([])
        return self._constraints.ub

    def obj_val(self, x: np.ndarray) -> float:
        assert x.shape == (self.num_vars,)

        return self._objective.eval(x)

    @property
    def constraints(self) -> Constraints:
        return self._constraints

    @property
    def obj(self) -> Objective:
        return self._objective

    def obj_grad(self, x: np.ndarray) -> np.ndarray:
        assert x.shape == (self.num_vars,)

        return self._objective.grad(x)

    def cons_val(self, x: np.ndarray) -> np.ndarray:
        if self._constraints is None:
            return np.array([])

        assert x.shape == (self.num_vars,)

        return self._constraints.eval(x)

    def cons_jac(self, x: np.ndarray) -> sp.sparse.spmatrix:
        if self._constraints is None:
            return sparse_zero((0, self.num_vars))

        assert x.shape == (self.num_vars,)

        return self._constraints.jac(x)

    def lag_hess(self, x: np.ndarray, lag: np.ndarray) -> sp.sparse.spmatrix:
        assert x.shape == (self.num_vars,)
        assert lag.shape == (self.num_cons,)

        obj_hess = self._objective.hess(x)

        if self._constraints is None:
            return obj_hess
        elif self._constraints.is_linear:
            return obj_hess

        cons_hess = self._constraints.hess(x, lag)

        return obj_hess + cons_hess
