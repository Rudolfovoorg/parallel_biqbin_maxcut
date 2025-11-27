from functools import wraps
from typing import Dict, Any
import scipy as sp
import numpy as np
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
        'shape': qubo_sparse_coo.shape,
        'nnz': qubo_sparse_coo.nnz,
        'row': qubo_sparse_coo.row.tolist(),
        'col': qubo_sparse_coo.col.tolist(),
        'data': qubo_sparse_coo.data.tolist()
    }


def from_sparse(sparse_matrix: dict) -> npt.NDArray[np.float32]:
    """Helper function that converts from sparse coo matrix to regular form

    Args:
        sparse_matrix (dict): scipy sparse coo matrix

    Returns:
        np.ndarray: regular form matrix
    """
    return sp.sparse.coo_matrix(
        (sparse_matrix['data'],
            (sparse_matrix['row'], sparse_matrix['col'])),
        shape=sparse_matrix['shape'], dtype='float'
    ).todense().getA()


def qubo_to_biqbin_representation(qubo) -> dict:
    """Converts a dense qubo represantation 2D array to the expected biqbin format of a json serializable 
    dict with 'qubo' key and a sparse qubo represantation as value.

    Args:
        qubo: 2D list or numpy array

    Returns:
        dict: json serializable dictionary that Biqbin can parse. Save to file and pass the path to DataGetterJson.
    """
    if not np.all(np.asarray(qubo) % 1 == 0):
        raise ValueError("All QUBO values need to be integers!")

    return {
        'qubo': to_sparse(qubo)
    }


def check_matrix_validity(input_matrix: np.ndarray) -> npt.NDArray[np.float64]:
    if not isinstance(input_matrix, np.ndarray):
        raise TypeError(
            f"Input matrix must be a numpy.ndarray, got {type(input_matrix)}")

    if not np.issubdtype(input_matrix.dtype, np.number):
        raise TypeError(
            f'Input matrix must use a numeric dtype (int or float), got {input_matrix.dtype}')

    if input_matrix.ndim != 2:
        raise ValueError(
            'Dimension of the input matrix needs to be 2!')

    n, m = input_matrix.shape
    if n != m:
        raise ValueError(
            f'Input matrix shape must be square (n, n), but got ({n}, {m})')

    adj_int = np.array(input_matrix, dtype=np.int64)
    if not np.all(input_matrix == adj_int):
        raise ValueError(
            'All values in the input matrix need to be integers!')

    return input_matrix


def check_matrix_validity_wrap(func):
    @wraps(func)
    def wrapper(*args, **kwargs) -> npt.NDArray[np.float64]:
        matrix_to_validate = func(*args, *kwargs)
        return check_matrix_validity(matrix_to_validate)
    
    return wrapper


"""
if not isinstance(maxcut_adjacency_matrix, np.ndarray):
    maxcut_adjacency_matrix = np.array(
        maxcut_adjacency_matrix, dtype=np.float64)
"""


def qubo_to_qplib_str(qubo: np.ndarray, problem_name: str) -> str:
    """Converts a dense qubo representation 2D np array into the qplib format string

    Args:
        qubo (np.ndarray): 2D list or numpy array
        problem_name (str): Saves the name of the problem inside the qplib format

    Returns:
        str: qplib format that can be saved directly to .qplib file or used elsewhere
    """
    qplib_data = problem_name + '\n'

    num_vars = qubo.shape[0]

    linear_terms = qubo.diagonal()
    num_linear = np.count_nonzero(linear_terms)

    quadratic_terms = sp.sparse.coo_matrix(qubo - np.diag(linear_terms))
    num_quadratic = quadratic_terms.nnz

    # qubo's are always QBN (Quadratic, Binary, No-constraints), minimization problems, as per qplib specification
    qplib_data += 'QBN\nminimize\n'
    qplib_data += f'{num_vars} # number of variables\n'
    qplib_data += f'{num_quadratic} # number of quadratic terms in the objective\n'

    # qplib format is row column value but lower triangular while ours is upper triangular so we swich columns and rows
    for i, j, v in zip(quadratic_terms.col, quadratic_terms.row, quadratic_terms.data):
        if v != 0:
            qplib_data += f'{i + 1} {j + 1} {v * 2}\n'

    qplib_data += f'0.0 # default value of linear coefficients in objective\n'
    qplib_data += f'{num_linear} # number of non-default linear coefficients in objective\n'
    for i, v in enumerate(linear_terms):
        if v != 0:
            qplib_data += f'{i + 1} {v}\n'

    qplib_data += '0.0 # objective constant\n'
    qplib_data += '1.79769313486232E+308 # value for infinity\n'
    qplib_data += '0.0 # default variable primal value in starting point\n'
    qplib_data += '0 # number of non-default variable primal values in starting point\n'
    qplib_data += '0.0 # default variable bound dual value in starting point\n'
    qplib_data += '0 # number of non-default variable bound dual values in starting point\n'
    qplib_data += '0 # number of non-default variable names\n'
    qplib_data += '0 # number of non-default constraint names\n'
    return qplib_data
