from typing import List
import os
import ctypes
import numpy as np
from biqbin_data_objects import BiqBinParameters


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
            BiqBinParameters
        ]
        self.biqbin.master_init.restype = int
        self.biqbin.master_main_loop.restype = int

        self.biqbin.worker_init.argtypes = [BiqBinParameters]
        self.biqbin.worker_init.restype = int
        self.biqbin.worker_main_loop.restype = int

        self.biqbin.getRank.restype = int

        # Initialize solver for evaluation
        self.biqbin.InitSolverWrapped.argtypes = [
            np.ctypeslib.ndpointer(
                dtype=np.float64,
                ndim=2,
                flags='C_CONTIGUOUS'
            ),
            ctypes.c_int,
            BiqBinParameters
        ]

        self.biqbin.setParams.argtypes = [BiqBinParameters]
        self.biqbin.setParams.restype = int

    # init both MPI and the solver, both need argc and argv, return rank if successful
    def initialize(self, graph_path, params_path) -> int:
        args = [b"./biqbin",
                graph_path.encode("utf-8"),
                params_path.encode("utf-8")
                ]
        argv = (ctypes.c_char_p * 3)(*args)

        rank = self.biqbin.initMPI(3, argv)

        if not self.biqbin.initSolver(3, argv) == 0:
            raise Exception("Invalid arguments")
        return rank

    # Initializes MPI in C, returns rank
    def init_MPI(self, graph_path, params_path) -> int:
        args = [b"./biqbin",
                graph_path.encode("utf-8"),
                params_path.encode("utf-8")
                ]
        argv = (ctypes.c_char_p * 3)(*args)
        rank = self.biqbin.initMPI(3, argv)
        return rank

    # Initialize the solver, parses graph, allocates global variables, ...
    def init_solver(self, graph_path, params_path) -> bool:
        args = [b"./biqbin",
                graph_path.encode("utf-8"),
                params_path.encode("utf-8")
                ]
        argv = (ctypes.c_char_p * 3)(*args)
        success = self.biqbin.initSolver(3, argv) == 0
        return success

    # master rank evaluates the root node and decides if further branching is needed
    def master_init(self, filename, L, num_verts, params) -> bool:
        # returns 0 if not over
        return self.biqbin.master_init(
            filename,
            L,
            num_verts,
            params
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
    def worker_main_loop(self):
        return self.biqbin.worker_main_loop() != 0

    # Frees memory in worker process, finalizes MPI
    def worker_end(self):
        self.biqbin.worker_end()
        self.biqbin.finalizeMPI()
