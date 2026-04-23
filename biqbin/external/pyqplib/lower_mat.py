import numpy as np
import scipy.sparse


class LowerMatrix:
    def __init__(self, dim, rows, cols, vals):
        assert rows.shape == cols.shape
        assert rows.shape == vals.shape
        assert (rows >= cols).all()
        assert rows.ndim == 1

        diag = np.where(rows == cols)[0]
        subdiag = np.where(rows != cols)[0]

        self.diag_vals = vals[diag]
        self.diag_rows = rows[diag]

        self.subdiag_rows = rows[subdiag]
        self.subdiag_cols = cols[subdiag]
        self.subdiag_vals = vals[subdiag]

        self.dim = dim

        self.shape = (dim, dim)

    def __bool__(self):
        return (self.subdiag_rows.size > 0) or (self.diag_rows.size > 0)

    def __repr__(self):
        dim = self.dim
        nnz = self.diag_rows.size + self.subdiag_rows.size
        return f"<lower part of {dim}x{dim} sparse matrix with {nnz} stored elements>"

    def dot(self, x):
        p = np.zeros_like(x)

        subdiag_rows = self.subdiag_rows
        subdiag_cols = self.subdiag_cols
        subdiag_vals = self.subdiag_vals

        for row, col, val in zip(subdiag_rows, subdiag_cols, subdiag_vals):
            p[row] += val * x[col]

        diag_vals = self.diag_vals
        diag_rows = self.diag_rows

        # Add diagonal
        p[diag_rows] += diag_vals * x[diag_rows]

        return p

    def full(self):
        all_entries = np.concatenate(
            [self.subdiag_vals, self.subdiag_vals, self.diag_vals]
        )

        all_rows = np.concatenate(
            [self.subdiag_rows, self.subdiag_cols, self.diag_rows]
        )

        all_cols = np.concatenate(
            [self.subdiag_cols, self.subdiag_rows, self.diag_rows]
        )

        return scipy.sparse.coo_matrix((all_entries, (all_rows, all_cols)), self.shape)
