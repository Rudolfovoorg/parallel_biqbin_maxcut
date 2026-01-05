class InitialValues:
    def __init__(self, primal, cons_dual, bound_dual):
        (self.num_vars,) = primal.shape
        (self.num_cons,) = cons_dual.shape

        assert bound_dual.shape == (self.num_vars,)

        self.primal = primal
        self.cons_dual = cons_dual
        self.bound_dual = bound_dual

    @property
    def x0(self):
        return self.primal

    @property
    def y0(self):
        return self.cons_dual

    @property
    def z0(self):
        return self.bound_dual
