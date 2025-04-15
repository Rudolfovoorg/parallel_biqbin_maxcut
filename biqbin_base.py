import os
import ctypes
import numpy as np
from biqbin_data_objects import _BabNode, _BiqBinParameters, _GlobalVariables, _Problem, _HeurState


class _BiqbinBase:
    def __init__(self):
        self._biqbin = ctypes.CDLL(os.path.abspath("biqbin.so"))
        # initialize MPI in C solver
        self._biqbin.initMPI.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p)
        ]
        # returns rank
        self._biqbin.initMPI.restype = int
        # start of master process initialization
        self._biqbin.master_init.argtypes = [
            ctypes.c_char_p,
            np.ctypeslib.ndpointer(
                dtype=np.float64,
                ndim=2,
                flags='C_CONTIGUOUS'
            ),
            ctypes.c_int,
            ctypes.c_int,
            _BiqBinParameters
        ]
        # end of master processes initialization
        self._biqbin.master_init_end.argtypes = [ctypes.POINTER(_BabNode)]
        self._biqbin.master_init_end.restype = ctypes.c_int

        # runs until all workers are idle
        self._biqbin.master_main_loop.restype = ctypes.c_int

        # worker initialization
        self._biqbin.worker_init.argtypes = [_BiqBinParameters]
        self._biqbin.worker_init.restype = ctypes.c_int

        # create new node
        self._biqbin.new_node.argtypes = [ctypes.POINTER(_BabNode)]
        self._biqbin.new_node.restype = ctypes.POINTER(_BabNode)

        # Worker main loop functions seperated
        # wait for over signal
        self._biqbin.worker_check_over.restype = ctypes.c_int

        # if got signal "not over" receive problem from another worker or master places into PQ
        self._biqbin.worker_receive_problem.argtypes = None
        self._biqbin.worker_receive_problem.restype = None

        # check time limit reached
        self._biqbin.time_limit_reached.restype = ctypes.c_int

        # Check if PQ empty then Pop node and evaluate
        self._biqbin.pq_is_empty.restype = ctypes.c_int
        # Return type of Bab_PQPop is a pointer to BabNode
        self._biqbin.pq_pop.restype = ctypes.POINTER(_BabNode)

        # get and set lower bound functions
        self._biqbin.get_lower_bound.restype = ctypes.c_double
        self._biqbin.set_lower_bound.argtypes = [ctypes.c_double]

        # after eval
        self._biqbin.after_evaluation.argtypes = [
            ctypes.POINTER(_BabNode),
            ctypes.c_double
        ]

        # Set params in C
        self._biqbin.setParams.argtypes = [_BiqBinParameters]
        self._biqbin.setParams.restype = ctypes.c_int

        # Get globals pointer stored in C solver
        self._biqbin.get_globals_pointer.restype = ctypes.POINTER(
            _GlobalVariables)

        # Unit test functions
        # create globals struct in c and return its pointer
        self._biqbin.init_globals.argtypes = [
            np.ctypeslib.ndpointer(
                dtype=np.float64,
                ndim=2,
                flags='C_CONTIGUOUS'
            ),
            ctypes.c_int
        ]
        self._biqbin.init_globals.restype = ctypes.POINTER(_GlobalVariables)
        # free globals struct
        self._biqbin.free_globals.argtypes = [ctypes.POINTER(_GlobalVariables)]
        # set random seed
        self._biqbin.srand.argtypes = [ctypes.c_int]

        # Split SDP_bound function
        # create and set globals->PP Problem
        self._biqbin.create_subproblem.argtypes = [
            ctypes.POINTER(_BabNode),  # current node
            ctypes.POINTER(_Problem),  # globals->SP
            ctypes.POINTER(_Problem)  # globals->PP
        ]
        # initialize SDPBound
        self._biqbin.init_sdp.argtypes = [
            ctypes.POINTER(_BabNode),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(_GlobalVariables)
        ]
        # prepares globals->Z instantiates the HeurState structure
        self._biqbin.heuristic_init.argtypes = [
            ctypes.POINTER(_Problem),  # globals->SP
            ctypes.POINTER(_Problem),  # globals->PP
            ctypes.POINTER(_BabNode),  # current node
            ctypes.POINTER(ctypes.c_double),  # globals->X
            ctypes.POINTER(ctypes.c_double)  # globals->Z
        ]
        self._biqbin.heuristic_init.restype = ctypes.POINTER(_HeurState)

        # preprocess for GW_heuristic
        self._biqbin.cholesky_factorization.argtypes = [
            ctypes.POINTER(_HeurState),
            ctypes.POINTER(ctypes.c_double)  # globals->Z
        ]
        # return info if we want to check in python if Cholesky factorization was successful
        self._biqbin.cholesky_factorization.restype = ctypes.c_int

        # postprocess after GW_heuristic
        self._biqbin.heuristic_postprocess.argtypes = [
            ctypes.POINTER(_HeurState),
            ctypes.POINTER(_BabNode),  # current node
            ctypes.POINTER(ctypes.c_int),  # *x solution vector
            ctypes.POINTER(ctypes.c_double),  # globals->X
            ctypes.POINTER(ctypes.c_double),  # globals->Z
            ctypes.c_double  # previous heuristics value
        ]
        # return 1 if new solution is better else 0
        self._biqbin.heuristic_postprocess.restype = ctypes.c_int

        # free HeurState memory, return lower bound
        self._biqbin.heuristic_finalize.argtypes = [ctypes.POINTER(_HeurState)]
        self._biqbin.heuristic_finalize.restype = ctypes.c_double

        # GW_heuristic is unchanged
        self._biqbin.GW_heuristic.argtypes = [
            ctypes.POINTER(_Problem),  # globals->SP
            ctypes.POINTER(_Problem),  # globals->PP
            ctypes.POINTER(_BabNode),  # current node
            ctypes.POINTER(ctypes.c_int),  # current solution vector
            ctypes.c_int,  # int num, unsure of its purpose, set to number of vertices
            ctypes.POINTER(ctypes.c_double)  # globals->Z
        ]
        self._biqbin.GW_heuristic.restype = ctypes.c_double

        # updates the best solution and lower bound in the solver if it is better than the previous one
        self._biqbin.update_solution_wrapped.argtypes = [
            ctypes.POINTER(_BabNode),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(_Problem)
        ]
        # SDPBound has a loop, this initializes it
        self._biqbin.init_main_sdp_loop.argtypes = [
            ctypes.POINTER(_GlobalVariables),
            ctypes.c_int
        ]
        self._biqbin.init_main_sdp_loop.restype = ctypes.c_int
        # Start of the loop, can call heuristics after
        self._biqbin.main_sdp_loop_start.argtypes = [
            ctypes.POINTER(_GlobalVariables)
        ]
        self._biqbin.main_sdp_loop_start.restype = ctypes.c_int
        # End of the loop, after the optional heuristics call
        self._biqbin.main_sdp_loop_end.argtypes = [
            ctypes.POINTER(_BabNode),
            ctypes.POINTER(_GlobalVariables)
        ]
        self._biqbin.main_sdp_loop_end.restype = ctypes.c_int
        # After SDP loop get the calculated upper bound and set in current node->upper_bound
        self._biqbin.get_upper_bound.argtypes = [
            ctypes.POINTER(_BabNode),
            ctypes.POINTER(_GlobalVariables)
        ]
        self._biqbin.get_upper_bound.restype = ctypes.c_double

        # Root node sets the globals->diff to be used by later evaluations
        self._biqbin.set_globals_diff.argtypes = [
            ctypes.POINTER(_GlobalVariables)
        ]
