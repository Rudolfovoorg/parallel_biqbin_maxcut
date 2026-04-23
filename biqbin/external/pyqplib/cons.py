from abc import ABC, abstractmethod

import numpy as np
import scipy as sp

from .lower_mat import LowerMatrix
from .util import sparse_zero


class Constraints(ABC):
    def __init__(self, lb: np.ndarray, ub: np.ndarray):
        self.lb = lb
        self.ub = ub

        assert lb.shape == ub.shape
        (self.num_cons,) = lb.shape

    @property
    @abstractmethod
    def is_linear(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def eval(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def jac(self, x: np.ndarray) -> sp.sparse.spmatrix:
        raise NotImplementedError()

    @abstractmethod
    def hess(self, x: np.ndarray, lag: np.ndarray) -> sp.sparse.spmatrix:
        raise NotImplementedError()


class LinearConstraints(Constraints):
    def __init__(self, jac, lb, ub):
        (self.num_cons, self.num_vars) = jac.shape
        self.mat = jac
        super().__init__(lb, ub)

    @property
    def is_linear(self):
        return True

    def eval(self, x):
        return self.mat.dot(x)

    def jac(self, x):
        return self.mat

    def hess(self, x, lag):
        num_vars = self.num_vars
        return sparse_zero((num_vars, num_vars))


class QuadraticConstraints(Constraints):
    def __init__(self, num_vars, hess, jac, lb, ub):
        self.hess_mats = [LowerMatrix(num_vars, *values) for values in hess]
        self.mat = jac
        super().__init__(lb, ub)

    @property
    def is_linear(self):
        return False

    def eval(self, x):
        c = self.mat.dot(x)

        for i, hess in self._hess_mats():
            c[i] += 0.5 * np.dot(x, hess.dot(x))

        return c

    def _hess_mats(self):
        return ((i, hess) for (i, hess) in enumerate(self.hess_mats) if hess)

    def jac(self, x):
        hess_rows = []
        hess_cols = []
        hess_vals = []

        for i, mat in self._hess_mats():
            p = mat.dot(x)
            nonzeros = np.where(p != 0)[0]

            if nonzeros.size == 0:
                continue

            nnz = nonzeros.size

            hess_cols.append(nonzeros)
            hess_rows.append(np.full((nnz,), i, dtype=int))
            hess_vals.append(p[nonzeros])

        if len(hess_rows) == 0:
            return self.mat

        hess_rows = np.concatenate(hess_rows)
        hess_cols = np.concatenate(hess_cols)
        hess_vals = np.concatenate(hess_vals)

        hess_mat = sp.sparse.coo_matrix(
            (hess_vals, (hess_rows, hess_cols)), self.mat.shape
        )

        return hess_mat + self.mat

    def hess(self, x, lag):
        return sum(lag[i] * mat.full() for (i, mat) in self._hess_mats())
