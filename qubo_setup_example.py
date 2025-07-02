import json
import numpy as np
import scipy as sp
from utils import qubo_to_biqbin_representation
    
# Some "qubo" 2D matrix, all values must be integers
qubo = np.array(
    [
        [-1, 3],
        [3, -1]
        ])

# Make a dictionary with the key "qubo" and to_sparse return value as value
qubo_dict = qubo_to_biqbin_representation(qubo)

# Save to disk, to be read as input instance file
with open("example_qubo.json", "w") as f:
    json.dump(qubo_dict, f)
    