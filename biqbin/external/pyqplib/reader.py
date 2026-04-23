import numpy as np
import scipy.sparse

from .cons import LinearConstraints, QuadraticConstraints
from .desc import ProblemDescription
from .init_vals import InitialValues
from .log import logger
from .obj import LinearObjective, QuadraticObjective
from .prob import Problem
from .types import (
    ProblemConsType,
    ProblemObjType,
    ProblemVarType,
    Sense,
    VarType,
)


def next_line(f):
    line = next(f)

    pos = line.find("#")

    if pos != -1:
        line = line[:pos]

    return line.strip()


def read_matrix(f, num_rows, num_cols):
    num_terms = int(next_line(f))

    assert num_terms >= 0

    rows = np.zeros((num_terms,), dtype=int)
    cols = np.zeros((num_terms,), dtype=int)
    entries = np.zeros((num_terms,), dtype=float)

    logger.debug(f"Reading matrix with {num_terms} nonzeros")

    for i in range(num_terms):
        row, col, entry = next_line(f).split()

        rows[i] = int(row)
        cols[i] = int(col)
        entries[i] = float(entry)

    rows -= 1
    cols -= 1

    assert (0 <= rows).all()
    assert (0 <= cols).all()

    assert (rows < num_rows).all()
    assert (cols < num_cols).all()

    return (rows, cols, entries)


def read_vec(f, dim, dtype=float):
    default_val = dtype(next_line(f))

    vec = np.full((dim,), default_val, dtype=dtype)

    num_other = int(next_line(f))

    logger.debug(f"Reading vector with {num_other} non-default values")

    for i in range(num_other):
        row, value = next_line(f).split()

        row = int(row) - 1
        value = dtype(value)

        assert 0 <= row < dim

        vec[row] = value

    return vec


def read_hess(f, num_vars):
    (rows, cols, entries) = read_matrix(f, num_vars, num_vars)

    assert (rows >= cols).all()

    return (rows, cols, entries)


def read_objective(sense, f, num_vars, obj_type):
    logger.debug("Reading objective")

    if obj_type == ProblemObjType.LINEAR:
        lin = read_vec(f, num_vars)
        offset = float(next_line(f))

        return LinearObjective(sense, lin, offset)

    (rows, cols, entries) = read_hess(f, num_vars)

    lin = read_vec(f, num_vars)

    offset = float(next_line(f))

    return QuadraticObjective(sense, rows, cols, entries, lin, offset)


def read_cons_quad(f, num_vars, num_cons):
    logger.debug("Reading quadratic constraints")
    num_quad = int(next_line(f))

    quad_cons = [(list(), list(), list()) for i in range(num_cons)]

    for i in range(num_quad):
        cons, row, col, entry = next_line(f).split()

        cons = int(cons) - 1
        row = int(row) - 1
        col = int(col) - 1
        entry = float(entry)

        assert 0 <= cons < num_cons
        assert 0 <= row < num_vars
        assert 0 <= col < num_vars

        quad_cons[cons][0].append(row)
        quad_cons[cons][1].append(col)
        quad_cons[cons][2].append(entry)

    quad_cons = [
        (
            np.array(quad_con[0], dtype=int),
            np.array(quad_con[1], dtype=int),
            np.array(quad_con[2], dtype=float),
        )
        for quad_con in quad_cons
    ]

    return quad_cons


def read_inf(f):
    inf = float(next_line(f))
    logger.debug(f"Value for inf: {inf}")
    return inf


def read_cons(f, num_vars, num_cons, cons_type):
    if num_cons == 0:
        inf = read_inf(f)
        return (inf, None)

    is_quad = cons_type in [ProblemConsType.CONVEX, ProblemConsType.GENERAL]

    if is_quad:
        hess = read_cons_quad(f, num_vars, num_cons)

    # linear terms
    (rows, cols, entries) = read_matrix(f, num_cons, num_vars)

    jac = scipy.sparse.coo_matrix(
        (entries, (rows, cols)), shape=(num_cons, num_vars))

    inf = read_inf(f)

    cons_lb = read_vec(f, num_cons)
    cons_ub = read_vec(f, num_cons)

    cons_ub[cons_ub >= inf] = np.inf
    cons_lb[cons_lb <= -inf] = -np.inf

    assert (cons_lb <= cons_ub).all()

    if is_quad:
        return (inf, QuadraticConstraints(num_vars, hess, jac, cons_lb, cons_ub))
    else:
        return (inf, LinearConstraints(jac, cons_lb, cons_ub))


def read_vartypes(f, var_type, var_lb, var_ub):
    (num_vars,) = var_lb.shape

    if var_type in [
        ProblemVarType.CONTINUOUS,
        ProblemVarType.BINARY,
        ProblemVarType.INTEGER,
    ]:
        var_type = {
            ProblemVarType.CONTINUOUS: VarType.CONTINUOUS,
            ProblemVarType.BINARY: VarType.BINARY,
            ProblemVarType.INTEGER: VarType.INTEGER,
        }[var_type]

        return np.full((num_vars,), var_type, dtype=object)

    logger.debug("Reading nonstandard variable types")

    (num_vars,) = var_lb.shape

    int_vars = read_vec(f, num_vars, dtype=int)

    var_types = np.empty(shape=(num_vars,), dtype=object)

    for i in range(num_vars):
        int_var = int_vars[i] == 1
        lb = var_lb[i]
        ub = var_ub[i]

        if not int_var:
            var_types[i] = VarType.CONTINUOUS
            continue

        binary = (lb == 0.0) and (ub == 1.0)

        if binary:
            var_types[i] = VarType.BINARY
        else:
            var_types[i] = VarType.INTEGER

    return var_types


def read_varbounds(f, num_vars, var_type, inf):
    logger.debug("Reading variable bounds")

    if var_type == ProblemVarType.BINARY:
        logger.debug("Binary variables, no bounds")
        var_lb = np.zeros((num_vars,))
        var_ub = np.full((num_vars,), 1.0)
        return (var_lb, var_ub)

    var_lb = read_vec(f, num_vars)
    var_ub = read_vec(f, num_vars)

    var_ub[var_ub >= inf] = np.inf
    var_lb[var_lb <= -inf] = -np.inf

    assert (var_lb <= var_ub).all()

    return (var_lb, var_ub)


def convert_var_flag(var_flag):
    if var_flag == "C":
        return ProblemVarType.CONTINUOUS
    elif var_flag == "B":
        return ProblemVarType.BINARY
    elif var_flag == "M":
        return ProblemVarType.MIXED_BINARY
    elif var_flag == "I":
        return ProblemVarType.INTEGER
    elif var_flag == "G":
        return ProblemVarType.GENERAL

    raise ValueError(f"Unknown variable flag {var_flag}")


def convert_obj_flag(obj_flag):
    if obj_flag == "L":
        return ProblemObjType.LINEAR
    elif obj_flag == "C":
        return ProblemObjType.CONVEX
    elif obj_flag == "Q":
        return ProblemObjType.GENERAL

    raise ValueError(f"Unknown objective flag {obj_flag}")


def convert_cons_flag(cons_flag):
    if cons_flag == "N":
        return ProblemConsType.UNCONSTRAINED
    elif cons_flag == "B":
        return ProblemConsType.BOXED
    elif cons_flag == "L":
        return ProblemConsType.LINEAR
    elif cons_flag == "C":
        return ProblemConsType.CONVEX
    elif cons_flag == "Q":
        return ProblemConsType.GENERAL

    raise ValueError(f"Unknown constraint flag {cons_flag}")


def convert_sense(sense):
    if sense == "maximize":
        return Sense.MAXIMIZE
    elif sense == "minimize":
        return Sense.MINIMIZE

    raise ValueError(f"Unknown sense {sense}")


def read_initial_values(f, num_vars, num_cons):
    logger.debug("Reading initial values")

    primal = read_vec(f, num_vars)

    if num_cons > 0:
        cons_dual = read_vec(f, num_cons)
    else:
        cons_dual = np.zeros((0,))

    var_dual = read_vec(f, num_vars)

    return InitialValues(primal, cons_dual, var_dual)


def read_description(filename) -> ProblemDescription:
    return _read_from(filename, read_problem=False) # pyright: ignore[reportReturnType]


def read_problem(filename) -> Problem:
    return _read_from(filename, read_problem=True) # pyright: ignore[reportReturnType]


def open_file(filename):
    import os
    import zipfile

    path, extension = os.path.splitext(filename)

    if extension == ".zip":
        basename = os.path.basename(path)
        return zipfile.Path(filename, at=basename).open("r")

    return open(filename, "r")


def _read_from(filename, read_problem):
    logger.info(f"Reading QPLIB instance from {filename}")

    with open_file(filename) as f:
        name = next_line(f)
        flags = next_line(f)
        sense = next_line(f)

        sense = convert_sense(sense)

        num_vars = int(next_line(f))

        obj_flag = flags[0]
        var_flag = flags[1]
        cons_flag = flags[2]

        logger.debug(
            f"Obj flag: {obj_flag}, var flag: {var_flag}, cons flag: {cons_flag}"
        )

        var_type = convert_var_flag(var_flag)
        obj_type = convert_obj_flag(obj_flag)
        cons_type = convert_cons_flag(cons_flag)

        if cons_type in [ProblemConsType.UNCONSTRAINED, ProblemConsType.BOXED]:
            num_cons = 0
        else:
            num_cons = int(next_line(f))

        description = ProblemDescription(
            name, filename, num_vars, num_cons, var_type, obj_type, cons_type
        )

        logger.debug(f"Num vars {num_vars}, num cons: {num_cons}")

        if not read_problem:
            return description

        obj = read_objective(sense, f, num_vars, obj_type)

        (inf, cons) = read_cons(f, num_vars, num_cons, cons_type)

        (var_lb, var_ub) = read_varbounds(f, num_vars, var_type, inf)

        var_types = read_vartypes(f, var_type, var_lb, var_ub)

        initial_values = read_initial_values(f, num_vars, num_cons)

        # Not occurring in any instance
        num_non_default_varnames = int(next_line(f))
        num_non_default_consnames = int(next_line(f))

        has_non_default_varnames = num_non_default_varnames != 0
        has_non_default_consnames = num_non_default_consnames != 0

        if has_non_default_consnames or has_non_default_varnames:
            raise ValueError("Non-standard variable / constraint names")

        finished = False

        try:
            next_line(f)
        except StopIteration:
            finished = True

        if not finished:
            raise ValueError("Trailing lines in input")

        return Problem(
            description, initial_values, obj, cons, var_lb, var_ub, var_types
        )


if __name__ == "__main__":
    import logging
    import sys

    logging.basicConfig(level=logging.INFO)

    for filename in sys.argv[1:]:
        logger.info(f"Reading {filename}")
        problem = read_problem(filename)
