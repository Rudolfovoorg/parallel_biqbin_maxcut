class ProblemDescription:
    def __init__(
        self, name, filename, num_vars, num_cons, var_type, obj_type, cons_type
    ):
        self.name = name
        self.filename = filename
        self.num_vars = num_vars
        self.num_cons = num_cons
        self.var_type = var_type
        self.obj_type = obj_type
        self.cons_type = cons_type
