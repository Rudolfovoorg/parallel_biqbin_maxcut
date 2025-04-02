import glob
import re
import ctypes
import numpy as np
from biqbin_data_objects import ParametersWrapper, _BiqBinParameters


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

    # marked for deletion
    def read_parameters_file(self, filename) -> _BiqBinParameters:
        params = ParametersWrapper()
        params.read_from_file(filename)
        return params.get_c_struct()

    def find_latest_output_file(self, input_file_path):
        # Match: input_path.output or input_path.output_<number>
        pattern = f"{input_file_path}.output*"
        candidates = glob.glob(pattern)

        if not candidates:
            return None

        def extract_suffix(file):
            # Match .output or .output_X
            match = re.match(re.escape(input_file_path) +
                             r"\.output(?:_(\d+))?$", file)
            if match:
                return int(match.group(1)) if match.group(1) else 0
            return -1  # fallback, shouldn't happen

        # Sort by suffix number
        candidates.sort(key=extract_suffix, reverse=True)
        return candidates[0]
