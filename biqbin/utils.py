import json
import time
from functools import wraps

import numpy as np
import scipy as sp
from numpy import typing as npt


def to_sparse(qubo: npt.ArrayLike):
    """Converts a 2D array to the expected sparse coo_matrix representation.

    Args:
        qubo: 2D list or numpy array

    Returns:
        dict: a sparse QUBO representation needed as input for the Solver. It is as a json serializable dict.
    """
    qubo_sparse_coo = sp.sparse.coo_matrix(qubo)
    return {
        "shape": qubo_sparse_coo.shape,
        "nnz": qubo_sparse_coo.nnz,
        "row": qubo_sparse_coo.row.tolist(),
        "col": qubo_sparse_coo.col.tolist(),
        "data": qubo_sparse_coo.data.tolist(),
    }


def from_sparse(sparse_matrix: dict) -> npt.NDArray[np.float64]:
    """Reconstruct a dense matrix from BiqBin's sparse dictionary format.

    Args:
        sparse_matrix (dict): Dictionary with `shape`, `row`, `col`, and `data`
            fields, as returned by `to_sparse`.

    Returns:
        np.ndarray: Dense floating-point matrix.
    """
    return sp.sparse.coo_matrix(
        (sparse_matrix["data"], (sparse_matrix["row"], sparse_matrix["col"])),
        shape=sparse_matrix["shape"],
        dtype="float",
    ).toarray()


def qubo_to_biqbin_representation(
    qubo: npt.ArrayLike, offset: float = 0.0, minimization: bool = True
) -> dict:
    """Converts a dense qubo represantation 2D array to the expected biqbin format of a json serializable
    dict with 'qubo' key and a sparse qubo represantation as value.

    Args:
        qubo (np.ndarray): 2D list or numpy array
        offset (float): an offset to be added to the solution value
        minimization (bool): objective sense

    Returns:
        dict: JSON-serializable problem dictionary. Save it to a file and read
            it with `biqbin.QuboFromJson`.

    Raises:
        ValueError: If any QUBO coefficient is not integer-valued.
    """
    if not np.all(np.asarray(qubo) % 1 == 0):
        raise ValueError("All QUBO values need to be integers!")

    return {"qubo": to_sparse(qubo), "offset": offset, "is_minimization": minimization}


def check_matrix_validity(input_matrix: np.ndarray) -> npt.NDArray[np.float64]:
    """Validate a square matrix with finite, integer-valued coefficients.

    Args:
        input_matrix (np.ndarray): Matrix with an integer or floating-point dtype.

    Returns:
        np.ndarray: The original matrix, without copying or changing its dtype.

    Raises:
        TypeError: If the input is not an ndarray with an integer or floating dtype.
        ValueError: If the input is not square and two-dimensional, or contains
            non-finite or non-integer values.
    """
    if not isinstance(input_matrix, np.ndarray):
        raise TypeError(f"Input matrix must be a numpy.ndarray, got {type(input_matrix)}")

    if not np.issubdtype(input_matrix.dtype, np.floating) and not np.issubdtype(
        input_matrix.dtype, np.integer
    ):
        raise TypeError(
            f"Input matrix must use a numeric dtype (int or float), got {input_matrix.dtype}"
        )

    if input_matrix.ndim != 2:
        raise ValueError("Dimension of the input matrix needs to be 2!")

    n, m = input_matrix.shape
    if n != m:
        raise ValueError(f"Input matrix shape must be square (n, n), but got ({n}, {m})")

    if not np.array_equal(input_matrix, np.round(input_matrix)) or not np.all(
        np.isfinite(input_matrix)
    ):
        raise ValueError(
            f"All values in the input matrix need to be integers!\nmatrix = \n{input_matrix}"
        )

    return input_matrix


def check_matrix_validity_wrap(func):
    """Validates np.ndarray returned by the wrapped function for the
    Biqbin solver.

    Functions runs first -> then validate.

    Checks for:
    - is a 2D matrix
    - is square matrix (n, n) shape
    - all integer values
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> npt.NDArray[np.float64]:
        value = func(*args, **kwargs)
        return check_matrix_validity(value)

    return wrapper


def check_matrix_validity_setter_wrap(setter):
    """Validates np.ndarray passed into the wrapped function
    Biqbin solver.

    Validate -> Then run the wrapped function.

    Checks for:
    - is a 2D matrix
    - is square matrix (n, n) shape
    - all integer values
    """

    @wraps(setter)
    def wrapper(self, value):
        value = check_matrix_validity(value)
        return setter(self, value)

    return wrapper


def data_collector(enabled_flag, data_box):
    """Collects heuristic data on root node if enabled

    Args:
        enabled_flag (str): Name of the boolean class attribute that enables collection.
        data_box (str): Name of the class attribute in which to store collected data.
    """

    def decorator(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            if not getattr(self, enabled_flag, False):
                return fn(self, *args, **kwargs)

            if getattr(self, data_box, None) is None:
                setattr(self, data_box, [])

            start = time.perf_counter()
            result = fn(self, *args, **kwargs)
            getattr(self, data_box).append(
                {
                    "time": time.perf_counter() - start,
                    "value": result,
                }
            )
            return result

        return wrapper

    return decorator


def convert_numpy_to_json_serializable(obj):
    """Convert a NumPy value for use as the `default` callback of `json.dump`.

    Args:
        obj (object): NumPy array or scalar that the JSON encoder cannot handle.

    Returns:
        object: A list for an ndarray, or the value returned by a NumPy scalar's
            `item()` method. The resulting value must itself be JSON-serializable.

    Raises:
        TypeError: If the object is not a NumPy array or scalar.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def divide_matrix_by_gcd(matrix: np.ndarray) -> int:
    """Takes in a ndarray matrix, finds the greatest common divisor and divides the values by it

    Args:
        matrix (np.ndarray): matrix to be divided in place

    Returns:
        int: greatest common divisor found
    """
    greatest_common_divisor = np.gcd.reduce(matrix.astype(int).flatten())
    if greatest_common_divisor > 1:
        # BZ: perform integer division so it works on np.int64 dtypes as well
        matrix //= greatest_common_divisor

    return int(greatest_common_divisor)


def flatten_dict(d: dict, prefix="", level=0):
    """Recursively flatten a dictionary, to create a pandas df from nested dicts

    Args:
        d (dict): data

    Yields:
        tuple[str, object]: A flattened key and its value.
    """
    for i, j in d.items():
        if isinstance(j, dict):
            yield from flatten_dict(j, f"{prefix}{i}_", level + 1)
        else:
            yield f"{prefix}{i}", j


def data_reader_pd(filelist):
    """Yield flattened records from JSON files for tabular analysis.

    This returns an iterator of dictionaries, not a DataFrame. To create one,
    pass the iterator to `pandas.DataFrame`; pandas is not required by this helper.

    Args:
        filelist (iterable): a list or equivalent of filepaths

    Yields:
        dict: flattenened dictionary
    """
    for filename in filelist:
        with open(filename) as f:
            yield dict(flatten_dict(json.load(f)))


def qubo_to_qplib_str(qubo: np.ndarray, offset: float, minimize: bool, problem_name: str):
    """Yield QPLIB text chunks for a dense QUBO matrix.

    Use `"".join(qubo_to_qplib_str(...))` to obtain one string, or pass the
    iterator to a file's `writelines()` method.

    Args:
        qubo (np.ndarray): Dense QUBO coefficient matrix.
        offset (float): Constant added to the objective value.
        minimize (bool): Whether to minimize rather than maximize the objective.
        problem_name (str): Problem name written into the QPLIB header.

    Yields:
        str: A QPLIB text chunk containing one or more lines.
    """
    # QPLIB expects a lower triangular matrix
    diag = np.diag(qubo)
    if np.array_equal(qubo, qubo.T):
        qubo = np.tril(qubo * 2)
    else:
        qubo = np.tril(qubo + qubo.T)

    np.fill_diagonal(qubo, diag)

    yield problem_name + "\n"

    num_vars = qubo.shape[0]

    linear_terms = qubo.diagonal()
    num_linear = np.count_nonzero(linear_terms)
    quadratic_terms = sp.sparse.coo_matrix(qubo)
    num_quadratic = quadratic_terms.nnz - num_linear

    min_max = "minimize\n" if minimize else "maximize\n"
    # qubo's are always QBN (Quadratic, Binary, No-constraints), minimization problems, as per qplib specification
    yield (
        "QBN\n"
        f"{min_max}"
        f"{num_vars} # number of variables\n"
        f"{num_quadratic} # number of quadratic terms in the objective\n"
    )

    for i, j, v in zip(quadratic_terms.row, quadratic_terms.col, quadratic_terms.data):
        if v != 0 and i != j:
            yield f"{i + 1} {j + 1} {v * 2}\n"

    yield (
        f"0.0 # default value of linear coefficients in objective\n"
        f"{num_linear} # number of non-default linear coefficients in objective\n"
    )

    for i, v in enumerate(linear_terms):
        if v != 0:
            yield f"{i + 1} {v}\n"

    yield f"{offset} # objective constant\n"

    yield (
        "1.79769313486232E+308 # value for infinity\n"
        "0.0 # default variable primal value in starting point\n"
        "0 # number of non-default variable primal values in starting point\n"
        "0.0 # default variable bound dual value in starting point\n"
        "0 # number of non-default variable bound dual values in starting point\n"
        "0 # number of non-default variable names\n"
        "0 # number of non-default constraint names\n"
    )


def save_qubo_problem_as_qplib(filename: str, problem):
    """Write a QUBO problem to QPLIB, overwriting an existing destination file.

    Args:
        filename (str): Destination path; parent directories must already exist.
        problem (biqbin.ProblemQubo): Problem supplying the coefficient matrix,
            objective offset and sense, and problem name.
    """
    with open(filename, "w") as f:
        f.writelines(
            qubo_to_qplib_str(
                problem.Q, problem.offset, problem.is_minimization, problem.problem_name
            )
        )


def save_qubo_problem_as_json(filename: str, problem):
    """Write a QUBO problem as BiqBin JSON, overwriting an existing file.

    Args:
        filename (str): Destination path; parent directories must already exist.
        problem (biqbin.ProblemQubo): Problem supplying the coefficient matrix,
            objective offset, and minimization flag.
    """
    with open(filename, "w") as f:
        json.dump(
            qubo_to_biqbin_representation(problem.Q, problem.offset, problem.is_minimization), f
        )


class PrettyPrint:
    """Render a class heading, aligned fields, and type-specific sections."""

    @staticmethod
    def _format_fields(fields: list[tuple[str, object]]) -> str:
        return "\n".join(f"{label:<21} = {value}" for label, value in fields)

    @staticmethod
    def _format_array(name: str, array: np.ndarray) -> str:
        return f"{name} = {array.shape}\n{array}"

    def _summary_fields(self) -> list[tuple[str, object]]:
        return []

    def _summary_sections(self, verbose: bool) -> list[str]:
        return []

    def format_summary(self, verbose: bool = False) -> str:
        """Return a class heading and summary; verbose adds conversion details.

        The heading shares the first line with a logging handler's prefix when
        an object is logged with ``logger.info("%s", object)``.
        """
        parts = [type(self).__name__]
        fields = self._summary_fields()
        if fields:
            parts.append(self._format_fields(fields))
        parts.extend(self._summary_sections(verbose))
        return "\n".join(parts)

    def __str__(self) -> str:
        return self.format_summary()

    def __repr__(self) -> str:
        return str(self)
