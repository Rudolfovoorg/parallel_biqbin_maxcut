from typing import List
import os
import ctypes
import numpy as np
from numpy.typing import NDArray
from biqbin_data_objects import BiqBinParameters, BabNode, GlobalVariables, Problem


class ParallelBiqbin:
    def __init__(self):
        self.biqbin = ctypes.CDLL(os.path.abspath("biqbin.so"))

        self.biqbin.initMPI.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p)
        ]
        self.biqbin.initMPI.restype = int

        self.biqbin.master_init.argtypes = [
            ctypes.c_char_p,
            np.ctypeslib.ndpointer(
                dtype=np.float64,
                ndim=2,
                flags='C_CONTIGUOUS'
            ),
            ctypes.c_int,
            ctypes.c_int,
            BiqBinParameters
        ]
        self.biqbin.master_init.restype = ctypes.c_int
        self.biqbin.master_main_loop.restype = ctypes.c_int

        self.biqbin.worker_init.argtypes = [BiqBinParameters]
        self.biqbin.worker_init.restype = ctypes.c_int

        # Worker main loop functions seperated
        # wait for over signal
        self.biqbin.worker_check_over.restype = ctypes.c_int

        # if got signal "not over" receive problem from another worker or master places into PQ
        self.biqbin.worker_receive_problem.argtypes = None
        self.biqbin.worker_receive_problem.restype = None

        # check time limit reached
        self.biqbin.time_limit_reached.argtypes = None
        self.biqbin.time_limit_reached.restype = ctypes.c_int

        # Check if PQ empty then Pop node and evaluate
        self.biqbin.isPQEmpty.restype = ctypes.c_int
        # Return type of Bab_PQPop is a pointer to BabNode
        self.biqbin.Bab_PQPop.restype = ctypes.POINTER(BabNode)

        # Save old lowerbound before evaluation
        self.biqbin.Bab_LBGet.restype = ctypes.c_double

        # Argument type for evaluate_node_wrapped is also pointer to BabNode
        self.biqbin.evaluate_node_wrapped.argtypes = [
            ctypes.POINTER(BabNode),
            ctypes.c_int
        ]
        self.biqbin.evaluate_node_wrapped.restype = None

        # after eval
        self.biqbin.after_evaluation.argtypes = [
            ctypes.POINTER(BabNode),
            ctypes.c_double
        ]

        # Set params in C
        self.biqbin.setParams.argtypes = [BiqBinParameters]
        self.biqbin.setParams.restype = ctypes.c_int

        # Unit test functions
        self.biqbin.get_globals.argtypes = [
            np.ctypeslib.ndpointer(
                dtype=np.float64,
                ndim=2,
                flags='C_CONTIGUOUS'
            ),
            ctypes.c_int
        ]
        self.biqbin.get_globals.restype = ctypes.POINTER(GlobalVariables)
        self.biqbin.freeMemory.argtypes = [ctypes.POINTER(GlobalVariables)]

        self.biqbin.Evaluate.argtypes = [
            ctypes.POINTER(BabNode),
            ctypes.POINTER(GlobalVariables),
            ctypes.c_int
        ]
        self.biqbin.Evaluate.restype = ctypes.c_double

        self.biqbin.srand.argtypes = [ctypes.c_int]
        self.biqbin.set_BabPbSize.argtypes = [ctypes.c_int]

    # Initializes MPI in C, returns rank
    def init_MPI(self, graph_path, params_path) -> int:
        args = [b"./biqbin",
                graph_path.encode("utf-8"),
                params_path.encode("utf-8")
                ]
        argv = (ctypes.c_char_p * 3)(*args)
        rank = self.biqbin.initMPI(3, argv)
        return rank

    # master rank evaluates the root node and decides if further branching is needed
    def master_init(self, filename, L: NDArray[np.float64], num_verts: int, num_edge: int, parameters: BiqBinParameters) -> bool:
        # returns 0 if not over
        return self.biqbin.master_init(
            filename,
            L,
            num_verts,
            num_edge,
            parameters
        ) != 0

    # Main loop for master rank, waits for communication from workers and responds until all are free
    def master_main_loop(self) -> bool:
        return self.biqbin.master_main_loop() != 0

    # Sends over signal to workers, frees memory in C, print to and close output file and finalize MPI
    def master_end(self):
        self.biqbin.master_end()
        self.biqbin.finalizeMPI()

    # workers first receive status if the solver is done, if not update the global lower bound
    def worker_init(self, params: BiqBinParameters) -> bool:
        return self.biqbin.worker_init(params) != 0

    # worker main loop in C, waits for either the over signal or babnode to process, branches and sends more nodes to other workers
    def worker_main_loop(self, rank: int) -> bool:
        # Wait for over signal
        over = self.biqbin.worker_check_over() != 0
        if over:
            return True
        # receive problem, insert it into priority queue in C
        self.biqbin.worker_receive_problem()

        # loops until worker becomes idle
        while self.biqbin.isPQEmpty() == 0:
            # check time limit
            if self.biqbin.time_limit_reached() != 0:
                return True

            # get node from PQ (heap.c)
            babnode = self.biqbin.Bab_PQPop()
            # Save previous g_lowerbound
            old_lb = self.biqbin.Bab_LBGet()
            # Evaluate Node
            self.biqbin.evaluate_node_wrapped(babnode, rank)
            # After eval, communicate new solution, frees node etc
            self.biqbin.after_evaluation(babnode, old_lb)
        # if PQ empty notify master of being idle
        self.biqbin.worker_send_idle()
        return False

    # Frees memory in worker process, finalizes MPI
    def worker_end(self):
        self.biqbin.worker_end()
        self.biqbin.finalizeMPI()

    # Unit test functions
    def set_random_seed(self, seed: int):
        self.biqbin.srand(seed)

    def set_parameters(self, params: BiqBinParameters):
        self.biqbin.setParams(params)

    def get_globals(self, L: NDArray[np.float64], num_verts: int):
        return self.biqbin.get_globals(L, num_verts)

    def free_globals(self, globals):
        self.biqbin.freeMemory(globals)

    def evaluate(self, node: BabNode, globals, rank: int):
        return self.biqbin.Evaluate(node, globals, rank)
