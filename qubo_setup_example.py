import json
import numpy as np
import scipy as sp

# 
def to_sparse(qubo: np.ndarray):
    """Converts a numpy 2D array to the expected sparse coo_matrix representation.

    Args:
        qubo (np.ndarray): 2D matrix

    Returns:
        dict: coo_matrix converted to list instead of numpy array for json serialization.
    """
    qubo_sparse_coo = sp.sparse.coo_matrix(qubo)
    return {
        'shape': qubo_sparse_coo.shape, 
        'nnz': qubo_sparse_coo.nnz, 
        'row': qubo_sparse_coo.row.tolist(), 
        'col': qubo_sparse_coo.col.tolist(), 
        'data': qubo_sparse_coo.data.tolist()
    }
    
# Some "qubo" 2D matrix, all values must be integers
qubo = np.array(
    [
        [-1, 3],
        [3, -1]
        ])

# Make a dictionary with the key "qubo" and to_sparse return value as value
qubo_dict = {
    "qubo": to_sparse(qubo)
}

# Save to disk, to be read as input instance file
with open("example_qubo.json", "w") as f:
    json.dump(qubo_dict, f)
    