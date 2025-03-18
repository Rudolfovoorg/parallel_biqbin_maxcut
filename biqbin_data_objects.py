import ctypes
import numpy as np


class BiqBinParameters(ctypes.Structure):
    """ creates a struct to match emxArray_real_T """

    _fields_ = [
        ('init_bundle_iter', ctypes.c_int),
        ('max_bundle_iter', ctypes.c_int),
        ('triag_iter', ctypes.c_int),
        ('pent_iter', ctypes.c_int),
        ('hept_iter', ctypes.c_int),
        ('max_outer_iter', ctypes.c_int),
        ('extra_iter', ctypes.c_int),
        ('violated_TriIneq', ctypes.c_double),
        ('TriIneq', ctypes.c_int),
        ('adjust_TriIneq', ctypes.c_int),
        ('PentIneq', ctypes.c_int),
        ('HeptaIneq', ctypes.c_int),
        ('Pent_Trials', ctypes.c_int),
        ('Hepta_Trials', ctypes.c_int),
        ('include_Pent', ctypes.c_int),
        ('include_Hepta', ctypes.c_int),
        ('root', ctypes.c_int),
        ('use_diff', ctypes.c_int),
        ('time_limit', ctypes.c_int),
        ('branchingStrategy', ctypes.c_int),
        ('detailed_output', ctypes.c_int),
    ]


NMAX = 1024


class BabSolution(ctypes.Structure):
    # Binary vector storing the solution
    _fields_ = [("X", ctypes.c_int * NMAX)]


class BabNode(ctypes.Structure):
    _fields_ = [
        # 0-1 vector specifying which nodes are fixed
        ("xfixed", ctypes.c_int * NMAX),
        ("sol", BabSolution),                 # 0-1 solution vector
        # Fractional vector from primal matrix
        ("fracsol", ctypes.c_double * NMAX),
        ("level", ctypes.c_int),              # Level (depth) in the B&B tree
        ("upper_bound", ctypes.c_double)      # Upper bound on max-cut solution
    ]

    def __lt__(self, other):
        return self.upper_bound < other.upper_bound
