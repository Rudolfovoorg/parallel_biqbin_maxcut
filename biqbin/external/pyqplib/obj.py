from abc import ABC, abstractmethod

import numpy as np
import scipy as sp

from .lower_mat import LowerMatrix
from .util import sparse_zero


class Objective(ABC):
    def __init__(self, sense):
        self.sense = sense

    @abstractmethod
    def eval(self, x: np.ndarray) -> float:
        raise NotImplementedError()

    @abstractmethod
    def grad(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def hess(self, x: np.ndarray) -> sp.sparse.spmatrix:
        raise NotImplementedError()


class LinearObjective(Objective):
    def __init__(self, sense, lin, offset):
        super().__init__(sense)
        self.lin = lin
        self.offset = offset

    def eval(self, x):
        return np.dot(self.lin, x) + self.offset

    def grad(self, x):
        return self.lin

    def hess(self, x):
        (num_vars,) = self.lin.shape

        return sparse_zero((num_vars, num_vars))


class QuadraticObjective(Objective):
    def __init__(self, sense, rows, cols, entries, grad, offset):
        super().__init__(sense)
        self.lin = grad
        self.offset = offset

        (num_vars,) = grad.shape

        self.mat = LowerMatrix(num_vars, rows, cols, entries)

    def eval(self, x):
        obj = 0.5 * np.dot(x, self.mat.dot(x))
        obj += np.dot(self.lin, x)
        obj += self.offset
        return obj

    def grad(self, x):
        return self.mat.dot(x) + self.lin

    def hess(self, x):
        return self.mat.full()
