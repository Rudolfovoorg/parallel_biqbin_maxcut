from functools import wraps
from typing import Tuple
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


def qubo_to_biqbin_representation(qubo, offset: float = 0.0, minimize: bool =  True) -> dict:
    """Converts a dense qubo represantation 2D array to the expected biqbin format of a json serializable 
    dict with 'qubo' key and a sparse qubo represantation as value.

    Args:
        qubo (np.ndarray): 2D list or numpy array
        offset (float): an offset to be added to the solution value

    Returns:
        dict: json serializable dictionary that Biqbin can parse. Save to file and pass the path to DataGetterJson.
    """
    if not np.all(np.asarray(qubo) % 1 == 0):
        raise ValueError("All QUBO values need to be integers!")

    return {
        'qubo': to_sparse(qubo),
        'offset': offset,
    }


def check_matrix_validity(input_matrix: np.ndarray) -> npt.NDArray[np.float64]:
    if not isinstance(input_matrix, np.ndarray):
        raise TypeError(
            f"Input matrix must be a numpy.ndarray, got {type(input_matrix)}")

    if not np.issubdtype(input_matrix.dtype, np.floating) and not np.issubdtype(input_matrix.dtype, np.integer):
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
            f'All values in the input matrix need to be integers!\nmatrix = \n{input_matrix}')

    return input_matrix

def check_matrix_validity_wrap(func):
    @wraps(func)
    def wrapper(*args, **kwargs) -> npt.NDArray[np.float64]:
        matrix_to_validate = func(*args, *kwargs)
        return check_matrix_validity(matrix_to_validate)

    return wrapper


def convert_numpy_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def divide_matrix_by_gcd(matrix: np.ndarray) -> int:
    greatest_common_divisor = np.gcd.reduce(matrix.astype(int).flatten())
    if greatest_common_divisor > 1:
        matrix /= greatest_common_divisor

    return int(greatest_common_divisor)