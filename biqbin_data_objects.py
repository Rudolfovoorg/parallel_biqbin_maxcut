import json
import os
import ctypes
import numpy as np


class BiqBinParameters:
    def __init__(self,
                 init_bundle_iter: int = 5,
                 max_bundle_iter: int = 15,
                 triag_iter: int = 5,
                 pent_iter: int = 5,
                 hept_iter: int = 5,
                 max_outer_iter: int = 20,
                 extra_iter: int = 10,
                 violated_TriIneq: float = 0.05,
                 TriIneq: int = 5000,
                 adjust_TriIneq: int = 1,
                 PentIneq: int = 5000,
                 HeptaIneq: int = 5000,
                 Pent_Trials: int = 60,
                 Hepta_Trials: int = 50,
                 include_Pent: int = 1,
                 include_Hepta: int = 1,
                 root: int = 0,
                 use_diff: int = 1,
                 time_limit: int = 0,
                 branchingStrategy: int = 1,
                 detailed_output: int = 0,
                 params_filepath: str = None
                 ):

        self.init_bundle_iter: int = init_bundle_iter
        self.max_bundle_iter: int = max_bundle_iter
        self.triag_iter: int = triag_iter
        self.pent_iter: int = pent_iter
        self.hept_iter: int = hept_iter
        self.max_outer_iter: int = max_outer_iter
        self.extra_iter: int = extra_iter
        self.violated_TriIneq: float = violated_TriIneq
        self.TriIneq: int = TriIneq
        self.adjust_TriIneq: int = adjust_TriIneq
        self.PentIneq: int = PentIneq
        self.HeptaIneq: int = HeptaIneq
        self.Pent_Trials: int = Pent_Trials
        self.Hepta_Trials: int = Hepta_Trials
        self.include_Pent: int = include_Pent
        self.include_Hepta: int = include_Hepta
        self.root: int = root
        self.use_diff: int = use_diff
        self.time_limit: int = time_limit
        self.branchingStrategy: int = branchingStrategy
        self.detailed_output: int = detailed_output

        if params_filepath is not None:
            self.read_from_file(params_filepath)

    def read_from_file(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileExistsError(f"{filepath} does not exist")

        with open(filepath, "r") as f:
            if ".json" in filepath:
                data = json.load(f)
            else:
                data = {}
                for line in f:
                    line = line.strip()
                    if not line or "=" not in line:
                        continue
                    key, value = map(str.strip, line.split("=", 1))
                    data[key] = value

            # Assign attributes from loaded data
            for key, value in data.items():
                if hasattr(self, key):
                    attr_type = type(getattr(self, key))

                    try:
                        if attr_type == int:
                            value = int(value)
                        elif attr_type == float:
                            value = float(value)
                    except ValueError:
                        continue

    def get_c_struct(self):
        return _BiqBinParameters(
            self.init_bundle_iter,
            self.max_bundle_iter,
            self.triag_iter,
            self.pent_iter,
            self.hept_iter,
            self.max_outer_iter,
            self.extra_iter,
            self.violated_TriIneq,
            self.TriIneq,
            self.adjust_TriIneq,
            self.PentIneq,
            self.HeptaIneq,
            self.Pent_Trials,
            self.Hepta_Trials,
            self.include_Pent,
            self.include_Hepta,
            self.root,
            self.use_diff,
            self.time_limit,
            self.branchingStrategy,
            self.detailed_output
        )


class _BiqBinParameters(ctypes.Structure):
    """ creates a struct to match """
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


NMAX = 1024  # Const currently


class _BabSolution(ctypes.Structure):
    _fields_ = [("X", ctypes.c_int * NMAX)]


class _BabNode(ctypes.Structure):
    _fields_ = [
        ("xfixed", ctypes.c_int * NMAX),
        ("sol", _BabSolution),
        ("fracsol", ctypes.c_double * NMAX),
        ("level", ctypes.c_int),
        ("upper_bound", ctypes.c_double)
    ]

    def __lt__(self, other):
        return self.upper_bound < other.upper_bound


class _Problem(ctypes.Structure):
    # typedef struct Problem {
    #     double *L;          // Objective matrix
    #     int n;              // size of L
    #     int NIneq;          // number of triangle inequalities
    #     int NPentIneq;      // number of pentagonal inequalities
    #     int NHeptaIneq;     // number of heptagonal inequalities
    #     int bundle;         // size of bundle
    # } Problem;
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
class _Triangle_Inequality(ctypes.Structure):
    _fields_ = [
        ("i", ctypes.c_int),
        ("j", ctypes.c_int),
        ("k", ctypes.c_int),
        ("type", ctypes.c_int),       # type: 1-4
        ("value", ctypes.c_double),   # cut violation
        ("y", ctypes.c_double),       # dual multiplier
    ]

# Pentagonal Inequality Structure


class _Pentagonal_Inequality(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),                           # type: 1-3
        ("permutation", ctypes.c_int * 5),                # array of 5 ints
        ("value", ctypes.c_double),                       # cut violation
        ("y", ctypes.c_double),                           # dual multiplier
    ]

# Heptagonal Inequality Structure


class _Heptagonal_Inequality(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),                           # type: 1-4
        ("permutation", ctypes.c_int * 7),                # array of 7 ints
        ("value", ctypes.c_double),                       # cut violation
        ("y", ctypes.c_double),                           # dual multiplier
    ]

# Global Variables Structure


class _GlobalVariables(ctypes.Structure):
    _fields_ = [
        # Problems
        ("SP", ctypes.POINTER(_Problem)),
        ("PP", ctypes.POINTER(_Problem)),
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
        ("Cuts", ctypes.POINTER(_Triangle_Inequality)),
        ("List", ctypes.POINTER(_Triangle_Inequality)),

        # Pentagonal inequalities
        ("Pent_Cuts", ctypes.POINTER(_Pentagonal_Inequality)),
        ("Pent_List", ctypes.POINTER(_Pentagonal_Inequality)),

        # Heptagonal inequalities
        ("Hepta_Cuts", ctypes.POINTER(_Heptagonal_Inequality)),
        ("Hepta_List", ctypes.POINTER(_Heptagonal_Inequality)),
    ]


class _HeurState(ctypes.Structure):
    _fields_ = [
        ("n", ctypes.c_int),
        ("N", ctypes.c_int),
        ("nn", ctypes.c_int),
        ("inc", ctypes.c_int),
        ("UPLO", ctypes.c_char),

        ("xh", ctypes.POINTER(ctypes.c_double)),
        ("fh", ctypes.c_double),
        ("temp_x", ctypes.POINTER(ctypes.c_int)),
    ]
