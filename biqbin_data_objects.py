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


# typedef struct Problem {
#     double *L;          // Objective matrix
#     int n;              // size of L
#     int NIneq;          // number of triangle inequalities
#     int NPentIneq;      // number of pentagonal inequalities
#     int NHeptaIneq;     // number of heptagonal inequalities
#     int bundle;         // size of bundle
# } Problem;

class Problem(ctypes.Structure):
    _fields_ = [
        ("L", ctypes.POINTER(ctypes.c_double)),
        ("n", ctypes.c_int),
        ("NIneq", ctypes.c_int),
        ("NPentIneq", ctypes.c_int),
        ("NHeptaIneq", ctypes.c_int),
        ("bundle", ctypes.c_int),
    ]


# typedef struct GlobalVariables {
#     int stopped;                    // true if the algorithm stopped at root node or after a time limit
#     double root_bound;                  // SDP upper bound at root node
#     double TIME;                        // CPU time
#     double diff;			            // difference between basic SDP relaxation and bound with added cutting planes
#     /********************************************************/


#     /********************************************************/
#     /*************** Specific to node ***********************/
#     /********************************************************/
#     /* PRIMAL variables */
#     double *X;                          // Stores current (psd) X (primal solution). Violated inequalities are computed from X.
#     double *Z;                          // Cholesky factorization: X = ZZ^T (used for heuristic)
#     double *X_bundle;                   // containts bundle matrices as columns
#     double *X_test;                     // matching pair X for gamma_test

#     /* DUAL variables */
#     double *dual_gamma;                      // (nonnegative) dual multiplier to cutting planes
#     double *dgamma;                     // step direction vector
#     double *gamma_test;
#     double *lambda;                     // vector containing scalars of convex combinations of bundle matrices X_i
#     double *eta;                        // dual multiplier to dual_gamma >= 0 constraint
#     double *F;                          // vector of values <L,X_i>
#     double *g;                          // subgradient
#     double *G;                          // matrix of subgradients

#     double f;                           // objective value of relaxation

#     /* Triangle Inequalities variables */
#     Triangle_Inequality *Cuts;          // vector (MaxTriIneqAdded) of current triangle inequality constraints
#     Triangle_Inequality *List;          // vector (params.TriIneq) of new violated triangle inequalities

#     /* Pentagonal Inequalities variables */
#     Pentagonal_Inequality *Pent_Cuts;   // vector (MaxPentIneqAdded) of current pentagonal inequality constraints
#     Pentagonal_Inequality *Pent_List;   // vector (params.PentIneq) of new violated pentagonal inequalities

#     /* Heptagonal Inequalities variables */
#     Heptagonal_Inequality *Hepta_Cuts;   // vector (MaxHeptaIneqAdded) of current heptagonal inequality constraints
#     Heptagonal_Inequality *Hepta_List;   // vector (params.HeptaIneq) of new violated heptagonal inequalities
# } GlobalVariables;

# Triangle Inequality Structure
class Triangle_Inequality(ctypes.Structure):
    _fields_ = [
        ("i", ctypes.c_int),
        ("j", ctypes.c_int),
        ("k", ctypes.c_int),
        ("type", ctypes.c_int),       # type: 1-4
        ("value", ctypes.c_double),   # cut violation
        ("y", ctypes.c_double),       # dual multiplier
    ]

# Pentagonal Inequality Structure


class Pentagonal_Inequality(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),                           # type: 1-3
        ("permutation", ctypes.c_int * 5),                # array of 5 ints
        ("value", ctypes.c_double),                       # cut violation
        ("y", ctypes.c_double),                           # dual multiplier
    ]

# Heptagonal Inequality Structure


class Heptagonal_Inequality(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),                           # type: 1-4
        ("permutation", ctypes.c_int * 7),                # array of 7 ints
        ("value", ctypes.c_double),                       # cut violation
        ("y", ctypes.c_double),                           # dual multiplier
    ]

# Global Variables Structure


class GlobalVariables(ctypes.Structure):
    _fields_ = [
        # Problems
        ("SP", ctypes.POINTER(Problem)),
        ("PP", ctypes.POINTER(Problem)),
        # General info
        ("stopped", ctypes.c_int),
        ("root_bound", ctypes.c_double),
        ("TIME", ctypes.c_double),
        ("diff", ctypes.c_double),

        # Primal variables
        ("X", ctypes.POINTER(ctypes.c_double)),
        ("Z", ctypes.POINTER(ctypes.c_double)),
        ("X_bundle", ctypes.POINTER(ctypes.c_double)),
        ("X_test", ctypes.POINTER(ctypes.c_double)),

        # Dual variables
        ("dual_gamma", ctypes.POINTER(ctypes.c_double)),
        ("dgamma", ctypes.POINTER(ctypes.c_double)),
        ("gamma_test", ctypes.POINTER(ctypes.c_double)),
        # Note the underscore to avoid Python keyword conflict
        ("lambda_", ctypes.POINTER(ctypes.c_double)),
        ("eta", ctypes.POINTER(ctypes.c_double)),
        ("F", ctypes.POINTER(ctypes.c_double)),
        ("g", ctypes.POINTER(ctypes.c_double)),
        ("G", ctypes.POINTER(ctypes.c_double)),

        ("f", ctypes.c_double),

        # Triangle inequalities
        ("Cuts", ctypes.POINTER(Triangle_Inequality)),
        ("List", ctypes.POINTER(Triangle_Inequality)),

        # Pentagonal inequalities
        ("Pent_Cuts", ctypes.POINTER(Pentagonal_Inequality)),
        ("Pent_List", ctypes.POINTER(Pentagonal_Inequality)),

        # Heptagonal inequalities
        ("Hepta_Cuts", ctypes.POINTER(Heptagonal_Inequality)),
        ("Hepta_List", ctypes.POINTER(Heptagonal_Inequality)),
    ]
