from typing import List
import os
import ctypes
import numpy as np
from numpy.typing import NDArray
from biqbin_data_objects import ParametersWrapper, GlobalVariables, _BiqBinParameters, _BabNode, Problem
from helper_functions import HelperFunctions


class ParallelBiqbin:
    def __init__(self, params: ParametersWrapper = ParametersWrapper()):
        self.params = params
        self.rank = None

        self.num_vertices: int = 0
        self._globals_p = None

        self.helper_functions = HelperFunctions()

        # C functions
        self.__biqbin = ctypes.CDLL(os.path.abspath("biqbin.so"))
        self.__biqbin.initMPI.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p)
        ]
        self.__biqbin.initMPI.restype = int

        self.__biqbin.master_init.argtypes = [
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
        self.__biqbin.master_init.restype = ctypes.c_int
        self.__biqbin.master_main_loop.restype = ctypes.c_int

        self.__biqbin.worker_init.argtypes = [_BiqBinParameters]
        self.__biqbin.worker_init.restype = ctypes.c_int

        # Worker main loop functions seperated
        # wait for over signal
        self.__biqbin.worker_check_over.restype = ctypes.c_int

        # if got signal "not over" receive problem from another worker or master places into PQ
        self.__biqbin.worker_receive_problem.argtypes = None
        self.__biqbin.worker_receive_problem.restype = None

        # check time limit reached
        self.__biqbin.time_limit_reached.restype = ctypes.c_int

        # Check if PQ empty then Pop node and evaluate
        self.__biqbin.isPQEmpty.restype = ctypes.c_int
        # Return type of Bab_PQPop is a pointer to BabNode
        self.__biqbin.Bab_PQPop.restype = ctypes.POINTER(_BabNode)

        # get and set lower bound functions
        self.__biqbin.Bab_LBGet.restype = ctypes.c_double
        self.__biqbin.set_lower_bound.argtypes = [ctypes.c_double]

        # Argument type for evaluate_node_wrapped is also pointer to BabNode
        self.__biqbin.evaluate_node_wrapped.argtypes = [
            ctypes.POINTER(_BabNode),
            ctypes.c_int
        ]
        self.__biqbin.evaluate_node_wrapped.restype = ctypes.c_double

        # after eval
        self.__biqbin.after_evaluation.argtypes = [
            ctypes.POINTER(_BabNode),
            ctypes.c_double
        ]

        # Set params in C
        self.__biqbin.setParams.argtypes = [_BiqBinParameters]
        self.__biqbin.setParams.restype = ctypes.c_int

        # Unit test functions
        self.__biqbin.get_globals.argtypes = [
            np.ctypeslib.ndpointer(
                dtype=np.float64,
                ndim=2,
                flags='C_CONTIGUOUS'
            ),
            ctypes.c_int
        ]
        self.__biqbin.get_globals_pointer.restype = ctypes.POINTER(
            GlobalVariables)
        self.__biqbin.get_globals.restype = ctypes.POINTER(GlobalVariables)
        self.__biqbin.free_globals.argtypes = [ctypes.POINTER(GlobalVariables)]

        self.__biqbin.Evaluate.argtypes = [
            ctypes.POINTER(_BabNode),
            ctypes.POINTER(GlobalVariables),
            ctypes.c_int
        ]
        self.__biqbin.Evaluate.restype = ctypes.c_double
        self.__biqbin.srand.argtypes = [ctypes.c_int]

        # Split SDP_bound function
        self.__biqbin.create_subproblem_wrapped.argtypes = [
            ctypes.POINTER(_BabNode)
        ]

        self.__biqbin.init_sdp.argtypes = [
            ctypes.POINTER(_BabNode),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(GlobalVariables)
        ]
        self.__biqbin.heuristics_wrapped.argtypes = [
            ctypes.POINTER(_BabNode),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(GlobalVariables)
        ]
        self.__biqbin.heuristics_wrapped.restype = ctypes.c_double

        self.__biqbin.update_solution_wrapped.argtypes = [
            ctypes.POINTER(_BabNode),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(GlobalVariables)
        ]

        self.__biqbin.init_main_sdp_loop.argtypes = [
            ctypes.POINTER(GlobalVariables),
            ctypes.c_int
        ]
        self.__biqbin.init_main_sdp_loop.restype = ctypes.c_int

        self.__biqbin.main_sdp_loop_start.argtypes = [
            ctypes.POINTER(GlobalVariables)
        ]
        self.__biqbin.main_sdp_loop_start.restype = ctypes.c_int

        self.__biqbin.main_sdp_loop_end.argtypes = [
            ctypes.POINTER(_BabNode),
            ctypes.POINTER(GlobalVariables)
        ]
        self.__biqbin.main_sdp_loop_end.restype = ctypes.c_int

        self.__biqbin.get_upper_bound.argtypes = [
            ctypes.POINTER(_BabNode),
            ctypes.POINTER(GlobalVariables)
        ]
        self.__biqbin.get_upper_bound.restype = ctypes.c_double

        self.__biqbin.set_globals_diff.argtypes = [
            ctypes.POINTER(GlobalVariables)
        ]

    def compute(self, graph_path):
        # init MPI in C, get rank
        self.rank = self.__init_MPI(graph_path)

        if self.rank == 0:
            # Only rank 0 needs input data about the graph, name to open output file, L matrix and num verts for the problem
            adj, num_verts, num_edge, name = self.helper_functions.read_maxcut_input(
                graph_path)
            self.num_vertices = num_verts
            L_matrix = self.helper_functions.get_SP_L_matrix(adj)
            # initialize master, if over == True don't go into main loop
            over = self.__master_init(
                name,
                L_matrix,
                num_verts,
                num_edge
            )
            while not over:
                over = self.__master_main_loop()
            # tell workers to end, release memory and finalize MPI
            self.__master_end()

        else:
            # Initialize solver for workers, needs params, master lets them know if is over
            over = self.__worker_init()
            while not over:
                over = self.__worker_main_loop()
            # Free memory and finalize MPI
            self.__worker_end()

    def get_lower_bound(self) -> float:
        return self.__biqbin.Bab_LBGet()

    def set_lower_bound(self, new_lb: float):
        self.__biqbin.set_lower_bound(new_lb)

    def evaluate_node(self, babnode) -> float:
        return self.__biqbin.evaluate_node_wrapped(babnode, self.rank)

    def time_limit_reached(self) -> bool:
        return self.__biqbin.time_limit_reached() == 1

    def pop_node(self) -> ctypes.POINTER:
        return self.__biqbin.Bab_PQPop()

    def is_pq_empty(self) -> bool:
        return self.__biqbin.isPQEmpty() != 0

    # worker main loop in C, waits for either the over signal or babnode to process, branches and sends more nodes to other workers
    def __worker_main_loop(self) -> bool:
        # Wait for over signal
        over = self.__biqbin.worker_check_over() != 0
        if over:
            return True
        # receive problem through MPI, insert it into priority queue in C
        self.__biqbin.worker_receive_problem()

        # loops until worker becomes idle
        while not self.is_pq_empty():
            # check time limit
            if self.params.time_limit == 1 and self.time_limit_reached():
                return True

            # get node from PQ (heap.c)
            babnode = self.pop_node()
            # Save previous g_lowerbound
            old_lb = self.get_lower_bound()
            # Evaluate Node
            self.__evaluate_node(babnode.contents)
            # After eval, communicate new solution, frees node etc
            self.__biqbin.after_evaluation(babnode, old_lb)

        # if PQ empty notify master of being idle
        self.__biqbin.worker_send_idle()
        return False

    # Initializes MPI in C, returns rank
    def __init_MPI(self, graph_path) -> int:
        args = [b"./biqbin",
                graph_path.encode("utf-8"),
                ]
        argv = (ctypes.c_char_p * 2)(*args)
        rank = self.__biqbin.initMPI(2, argv)
        return rank

    # master rank evaluates the root node and decides if further branching is needed
    def __master_init(self, filename, L: NDArray[np.float64], num_verts: int, num_edge: int) -> bool:
        # returns 0 if not over
        over = self.__biqbin.master_init(
            filename,
            L,
            num_verts,
            num_edge,
            self.params.get_c_struct()
        ) != 0
        self._globals_p = self.__biqbin.get_globals_pointer()
        return over

    # Main loop for master rank, waits for communication from workers and responds until all are free
    def __master_main_loop(self) -> bool:
        return self.__biqbin.master_main_loop() != 0

    # Sends over signal to workers, frees memory in C, print to and close output file and finalize MPI
    def __master_end(self):
        self.__biqbin.master_end()
        self.__biqbin.finalizeMPI()

    # workers first receive status if the solver is done, if not update the global lower bound
    def __worker_init(self) -> bool:
        over = self.__biqbin.worker_init(self.params.get_c_struct()) != 0
        self._globals_p = self.__biqbin.get_globals_pointer()
        self.num_vertices = self._globals_p.contents.SP.contents.n - 1
        return over

    # Frees memory in worker process, finalizes MPI
    def __worker_end(self):
        self.__biqbin.worker_end()
        self.__biqbin.finalizeMPI()

    ##############################################################
    ##################### evaluate seperatered ###################
    ##############################################################

    def __evaluate_node(self, babnode):
        # changes globals->PP using globals->SP and current BabNode
        self.__create_subproblem(babnode)
        # Stores the best solution for node, updates it in BabNode x[BabPbSize] local variable in SDPbound function
        sol_x = (ctypes.c_int * (self.num_vertices - 1))()
        # update solution_vector_x for the given subproblem and node
        self.__biqbin.init_sdp(babnode, sol_x, self._globals_p)
        # Run heuristics and update solution first time
        self.run_heuristics(babnode, sol_x)
        self.__update_solution_wrapped(babnode, sol_x)

        is_root = 1 if self.rank == 0 else 0  # runs differently on root node
        over = self.__biqbin.init_main_sdp_loop(
            self._globals_p,
            is_root
        )
        while not over:
            prune = self.__biqbin.main_sdp_loop_start(self._globals_p)
            if not prune:
                for i in range(self.num_vertices - 1):
                    if babnode.xfixed[i]:
                        sol_x[i] = babnode.sol.X[i]
                    else:
                        sol_x[i] = 0
                self.run_heuristics(babnode, sol_x)
                self.__update_solution_wrapped(babnode, sol_x)

            over = self.__biqbin.main_sdp_loop_end(
                babnode, self._globals_p
            )

        if self.rank == 0 and self.params.use_diff == 1:
            self.__biqbin.set_globals_diff(self._globals_p)
        return self.__biqbin.get_upper_bound(babnode, self._globals_p)

    def __create_subproblem(self, babnode):
        self.__biqbin.create_subproblem_wrapped(babnode, self._globals_p)

    # should only need subproblem Laplacean and subproblem size
    def run_heuristics(self, babnode, sol_x) -> float:
        # returns lower bound for the given node and subproblem
        return self.__biqbin.heuristics_wrapped(ctypes.pointer(babnode), sol_x, self._globals_p)

    def __update_solution_wrapped(self, babnode, sol_x):
        # if the heuristic solution (lower bound) in *x is better than the old one, it will update it in C
        self.__biqbin.update_solution_wrapped(babnode, sol_x, self._globals_p)

    ##############################################################
    ##################### Unit test functions ####################
    ##############################################################

    def set_random_seed(self, seed: int):
        self.__biqbin.srand(seed)

    def set_parameters(self, params: ParametersWrapper):
        self.params = params
        self.__biqbin.setParams(params.get_c_struct())

    def get_globals(self, L: NDArray[np.float64], num_verts: int):
        self.num_vertices = num_verts
        return self.__biqbin.get_globals(L, num_verts)

    def free_globals(self, globals):
        self.__biqbin.free_globals(globals)

    def evaluate(self, node: _BabNode, globals, rank: int):
        # return self.__biqbin.Evaluate(node, globals, rank)
        self._globals_p = globals
        self.rank = rank
        return self.__evaluate_node(node)
