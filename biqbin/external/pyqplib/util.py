import scipy.sparse


def sparse_zero(shape):
    return scipy.sparse.coo_matrix(([], ([], [])), shape)
