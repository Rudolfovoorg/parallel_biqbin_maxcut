import scipy as sp
import numpy as np

def to_sparse(qubo):
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