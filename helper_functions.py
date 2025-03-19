import ctypes
import numpy as np
from biqbin_data_objects import BiqBinParameters


class HelperFunctions:
    def read_maxcut_input(self, filename):
        with open(filename, 'r') as f:
            # Read number of vertices and edges
            num_vertices, num_edges = map(int, f.readline().split())

            # Allocate adjacency matrix as a contiguous array (row-major)
            adj_matrix = np.zeros(
                (num_vertices, num_vertices), dtype=np.float64)

            # Read edges
            for _ in range(num_edges):
                i, j, weight = f.readline().split()
                i, j = int(i) - 1, int(j) - 1  # Convert to zero-based indexing
                weight = float(weight)

                adj_matrix[i, j] = weight
                adj_matrix[j, i] = weight  # Since the graph is undirected

            # Create the MaxCutInputData struct
            name = filename.encode('utf-8')

            # Returning the matrix for debugging
            return adj_matrix, num_vertices, num_edges, name

    # Construct the L matrix, which is [ Laplacian,  Laplacian*e; (Laplacian*e)',  e'*Laplacian*e]
    def get_SP_L_matrix(self, Adj):
        num_vertices = Adj.shape[0]

        # Initialize Laplacian matrix L
        L = np.zeros((num_vertices, num_vertices))

        # Compute Adje = Adj * e (sum of each row)
        Adje = Adj.sum(axis=1)

        # Compute Diag(Adje) - diagonal matrix with Adje values
        tmp = np.diag(Adje)

        # Construct Laplacian matrix: L = tmp - Adj, excluding the last row and column
        L[:-1, :-1] = tmp[:-1, :-1] - Adj[:-1, :-1]

        # Compute vector parts and constant part
        sum_row = L[:-1, :-1].sum(axis=1)
        sum_total = sum_row.sum()

        # Fill vector parts and constant term
        L[:-1, -1] = sum_row
        L[-1, :-1] = sum_row
        L[-1, -1] = sum_total

        return L

    def read_parameters_file(self, filename):
        params = BiqBinParameters()
        # Mapping field names to types
        field_types = {name: typ for name, typ in BiqBinParameters._fields_}
        with open(filename, 'r') as f:
            for line in f.readlines():
                key_val = line.strip().split('=')
                if len(key_val) != 2:
                    print(f"Skipping invalid line: {line.strip()}")
                    continue  # Skip invalid lines
                key, value = key_val[0].strip(), key_val[1].strip()
                if hasattr(params, key):
                    field_type = field_types[key]
                    # Convert value to the correct type
                    if field_type == ctypes.c_int:
                        setattr(params, key, int(value))
                    elif field_type == ctypes.c_double:
                        setattr(params, key, float(value))
                    else:
                        print(f"Unknown type for field: {key}")
                else:
                    print(f"Unknown parameter: {key}")
        return params
