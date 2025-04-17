import os
from abc import ABC, abstractmethod
import ctypes
import numpy as np
from biqbin_data_objects import _BabNode, BiqBinParameters, _BiqBinParameters, _GlobalVariables, _Problem, _HeurState


class _BiqbinBase(ABC):
    def __init__(self):
        self.__biqbin = ctypes.CDLL(os.path.abspath("biqbin.so"))
        # initialize MPI in C solver, returns rank
        self.__biqbin.init_mpi_wrapped.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p)
        ]
        self.__biqbin.init_mpi_wrapped.restype = int

        # start of master process initialization
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
        self.__biqbin.master_init.restype = None

        # end of master processes initialization
        self.__biqbin.master_init_end.argtypes = [ctypes.POINTER(_BabNode)]
        self.__biqbin.master_init_end.restype = ctypes.c_int

        # runs until all workers are idle
        self.__biqbin.master_main_loop.argtypes = []
        self.__biqbin.master_main_loop.restype = ctypes.c_int

        # worker initialization
        self.__biqbin.worker_init.argtypes = [_BiqBinParameters]
        self.__biqbin.worker_init.restype = ctypes.c_int

        # create new node
        self.__biqbin.new_node.argtypes = [ctypes.POINTER(_BabNode)]
        self.__biqbin.new_node.restype = ctypes.POINTER(_BabNode)

        # Worker main loop functions seperated
        # wait for over signal
        self.__biqbin.worker_check_over.argtypes = []
        self.__biqbin.worker_check_over.restype = ctypes.c_int

        # if got signal "not over" receive problem from another worker or master places into PQ
        self.__biqbin.worker_receive_problem.argtypes = []
        self.__biqbin.worker_receive_problem.restype = None

        # check time limit reached
        self.__biqbin.time_limit_reached.argtypes = []
        self.__biqbin.time_limit_reached.restype = ctypes.c_int

        # Check if PQ empty then Pop node and evaluate
        self.__biqbin.pq_is_empty.argtypes = []
        self.__biqbin.pq_is_empty.restype = ctypes.c_int
        # Return type of Bab_PQPop is a pointer to BabNode
        self.__biqbin.pq_pop.restype = ctypes.POINTER(_BabNode)

        # get and set lower bound functions
        self.__biqbin.get_lower_bound.argtypes = []
        self.__biqbin.get_lower_bound.restype = ctypes.c_double

        self.__biqbin.set_lower_bound.argtypes = [ctypes.c_double]
        self.__biqbin.set_lower_bound.restype = None

        # after eval
        self.__biqbin.after_evaluation.argtypes = [
            ctypes.POINTER(_BabNode),
            ctypes.c_double
        ]
        self.__biqbin.after_evaluation.restype = None

        # Set params in C
        self.__biqbin.set_parameters.argtypes = [_BiqBinParameters]
        self.__biqbin.set_parameters.restype = ctypes.c_int

        # Get globals pointer stored in C solver
        self.__biqbin.get_globals_pointer.argtypes = []
        self.__biqbin.get_globals_pointer.restype = ctypes.POINTER(
            _GlobalVariables)

        # Unit test functions
        # create globals struct in c and return its pointer
        self.__biqbin.init_globals.argtypes = [
            np.ctypeslib.ndpointer(
                dtype=np.float64,
                ndim=2,
                flags='C_CONTIGUOUS'
            ),
            ctypes.c_int
        ]
        self.__biqbin.init_globals.restype = ctypes.POINTER(_GlobalVariables)

        # free globals struct
        self.__biqbin.free_globals.argtypes = [
            ctypes.POINTER(_GlobalVariables)
        ]
        self.__biqbin.free_globals.restype = None

        # set random seed
        self.__biqbin.srand.argtypes = [ctypes.c_int]
        self.__biqbin.srand.restype = None

        # Split SDP_bound function
        # create and set globals->PP Problem
        self.__biqbin.create_subproblem.argtypes = [
            ctypes.POINTER(_BabNode),  # current node
            ctypes.POINTER(_Problem),  # globals->SP
            ctypes.POINTER(_Problem)  # globals->PP
        ]
        self.__biqbin.create_subproblem.restype = None

        # initialize SDPBound
        self.__biqbin.init_sdp.argtypes = [
            ctypes.POINTER(_BabNode),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(_GlobalVariables)
        ]
        self.__biqbin.init_sdp.restype = None
        # prepares globals->Z instantiates the HeurState structure
        self.__biqbin.heuristic_init.argtypes = [
            ctypes.POINTER(_Problem),  # globals->SP
            ctypes.POINTER(_Problem),  # globals->PP
            ctypes.POINTER(_BabNode),  # current node
            ctypes.POINTER(ctypes.c_double),  # globals->X
            ctypes.POINTER(ctypes.c_double)  # globals->Z
        ]
        self.__biqbin.heuristic_init.restype = ctypes.POINTER(_HeurState)

        # preprocess for GW_heuristic
        self.__biqbin.cholesky_factorization.argtypes = [
            ctypes.POINTER(_HeurState),
            ctypes.POINTER(ctypes.c_double)  # globals->Z
        ]
        # return info if we want to check in python if Cholesky factorization was successful
        self.__biqbin.cholesky_factorization.restype = ctypes.c_int

        # postprocess after GW_heuristic
        self.__biqbin.heuristic_postprocess.argtypes = [
            ctypes.POINTER(_HeurState),
            ctypes.POINTER(_BabNode),  # current node
            ctypes.POINTER(ctypes.c_int),  # *x solution vector
            ctypes.POINTER(ctypes.c_double),  # globals->X
            ctypes.POINTER(ctypes.c_double),  # globals->Z
            ctypes.c_double  # previous heuristics value
        ]
        # return 1 if new solution is better else 0
        self.__biqbin.heuristic_postprocess.restype = ctypes.c_int

        # free HeurState memory, return lower bound
        self.__biqbin.heuristic_finalize.argtypes = [
            ctypes.POINTER(_HeurState)]
        self.__biqbin.heuristic_finalize.restype = ctypes.c_double

        # GW_heuristic is unchanged
        self.__biqbin.GW_heuristic.argtypes = [
            ctypes.POINTER(_Problem),  # globals->SP
            ctypes.POINTER(_Problem),  # globals->PP
            ctypes.POINTER(_BabNode),  # current node
            ctypes.POINTER(ctypes.c_int),  # current solution vector
            ctypes.c_int,  # int num, unsure of its purpose, set to number of vertices
            ctypes.POINTER(ctypes.c_double)  # globals->Z
        ]
        self.__biqbin.GW_heuristic.restype = ctypes.c_double

        # updates the best solution and lower bound in the solver if it is better than the previous one
        self.__biqbin.update_solution_wrapped.argtypes = [
            ctypes.POINTER(_BabNode),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(_Problem)
        ]
        self.__biqbin.update_solution_wrapped.restype = None
        # SDPBound has a loop, this initializes it
        self.__biqbin.init_main_sdp_loop.argtypes = [
            ctypes.POINTER(_GlobalVariables),
            ctypes.c_int
        ]
        self.__biqbin.init_main_sdp_loop.restype = ctypes.c_int

        # Start of the loop, can call heuristics after
        self.__biqbin.main_sdp_loop_start.argtypes = [
            ctypes.POINTER(_GlobalVariables)
        ]
        self.__biqbin.main_sdp_loop_start.restype = ctypes.c_int

        # End of the loop, after the optional heuristics call
        self.__biqbin.main_sdp_loop_end.argtypes = [
            ctypes.POINTER(_BabNode),
            ctypes.POINTER(_GlobalVariables)
        ]
        self.__biqbin.main_sdp_loop_end.restype = ctypes.c_int

        # After SDP loop get the calculated upper bound and set in current node->upper_bound
        self.__biqbin.get_upper_bound.argtypes = [
            ctypes.POINTER(_BabNode),
            ctypes.POINTER(_GlobalVariables)
        ]
        self.__biqbin.get_upper_bound.restype = ctypes.c_double

        # Root node sets the globals->diff to be used by later evaluations
        self.__biqbin.set_globals_diff.argtypes = [
            ctypes.POINTER(_GlobalVariables)
        ]

        # Notify workers that solving is over, print output, free memory
        self.__biqbin.master_end.argtypes = []
        self.__biqbin.master_end.restype = None

        # Free memory
        self.__biqbin.worker_end.argtypes = []
        self.__biqbin.worker_end.restype = None

        # finalize mpi
        self.__biqbin.MPI_Finalize.argtypes = []
        self.__biqbin.MPI_Finalize.restype = None

        # Notify master rank that this process has nothing pq
        self.__biqbin.worker_send_idle.argtypes = []
        self.__biqbin.worker_send_idle.restype = None

        # Increases the number of nodes evaluated for final output
        self.__biqbin.increase_num_eval_nodes.argtypes = []
        self.__biqbin.increase_num_eval_nodes.restype = None

    @abstractmethod
    def compute(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute_lower_bound(self, _Babnode, sol_x):
        pass

    @abstractmethod
    def compute_upper_bound(self, _Babnode):
        pass

    ###########################################################
    ################### COMMON FUNCTIONS ######################
    ###########################################################
    def _init_mpi(self, argc: int, argv: list[str]) -> int:
        """Initialize MPI in C solver

        Args:
            argc (int): number of CLI arguments
            argv (list[str]): CLI arguments

        Returns:
            int: rank
        """
        argv_c = (ctypes.c_char_p * len(argv))(*[s.encode() for s in argv])
        return self.__biqbin.init_mpi_wrapped(argc, argv_c)

    def _get_globals_pointer(self) -> ctypes._Pointer:
        """Get the pointer to the GlobalVariables *globals that have been initialized in C solver

        Returns:
            _Pointer: pointer to the globals structure in C
        """
        return self.__biqbin.get_globals_pointer()

    ###########################################################
    #################### MASTER PROCESS #######################
    ###########################################################
    def _master_init_start(self, name: str, L: np.ndarray, num_vertices: int, num_edges: int, params: BiqBinParameters):
        """Sets globals, heap, prints initial output, communicates main problem SP to worker processes

        Args:
            name (str): graph_path for the creation of output file
            L (np.ndarray): Laplacian matrix created for SP->L
            num_vertices (int): number of vertices in the graph
            num_edges (int): number of edges in the graph, needed only for the output print
            params (_BiqBinParameters): biqbin parameters structure
        """
        self.__biqbin.master_init(
            name.encode("utf-8"),
            L,
            num_vertices,
            num_edges,
            params.get_c_struct()
        )

    def _master_init_end(self, node: _BabNode) -> bool:
        """Branch or prune from evaluated root node, communicate with worker processes

        Args:
            node (_BabNode): root node, after evaluation

        Returns:
            bool: True if done solving
        """
        return self.__biqbin.master_init_end(ctypes.pointer(node)) != 0

    def _master_main_loop(self) -> bool:
        """Communicates with worker processes

        Returns:
            bool: True if done solving
        """
        return self.__biqbin.master_main_loop() != 0

    def _master_end(self):
        """Sends over signal to workers, frees memory in C, print to output file and close it, finalize MPI
        """
        self.__biqbin.master_end()
        self.__biqbin.MPI_Finalize()

    ###########################################################
    #################### WORKER PROCESS #######################
    ###########################################################
    def _worker_init(self, params: BiqBinParameters) -> bool:
        """Initialize solver in worker processes (rank != 0)

        Args:
            params (BiqBinParameters): parameters helper class

        Returns:
            bool: True if done solving
        """
        return self.__biqbin.worker_init(params.get_c_struct()) != 0

    def _worker_check_over(self) -> bool:
        """Receive "over" message from any source through MPI

        Returns:
            bool: True if received solving is over.
        """
        return self.__biqbin.worker_check_over() != 0

    def _worker_receive_problem(self):
        """Receive global lower bound and BabNode, pushes it to it's local pq to evaluate in the loop
        """
        self.__biqbin.worker_receive_problem()

    def _after_evaluation(self, node: _BabNode, old_lb: float):
        """Update masters global lowerbound if found better, prune node or branch and distribute to other workers

        Args:
            node (_BabNode): evaluated node
            old_lb (float): previous lover bound (before node evaluation)
        """
        self.__biqbin.after_evaluation(ctypes.pointer(node), old_lb)

    def _worker_send_idle(self):
        """Send IDLE status to master process to notify this process is free
        """
        self.__biqbin.worker_send_idle()

    def _worker_end(self):
        """Frees memory in worker process, finalizes MPI
        """
        self.__biqbin.worker_end()
        self.__biqbin.MPI_Finalize()

    def _time_limit_reached(self) -> bool:
        """check in C solver if time limit is reached, if yes end worker main loop

        Returns:
            bool: True if out of time
        """
        return self.__biqbin.time_limit_reached() == 1

    ###########################################################
    #################### heap.c FUNCTIONS  ####################
    ###########################################################

    def _pq_pop(self) -> _BabNode:
        """Pop node from the priority queue in heap.c

        Returns:
            _BabNode(ctype.Structure): defined in biqbin_data_objects.py
        """
        return self.__biqbin.pq_pop().contents

    def _pq_is_empty(self) -> bool:
        """Check priority queue in heap.c if empty
        Returns:
            bool: True if PQ in heap.c is empty
        """
        return self.__biqbin.pq_is_empty() != 0

    def _increase_num_eval_nodes(self):
        """Increase the number of evaluated nodes
        """
        self.__biqbin.increase_num_eval_nodes()

    def _get_lower_bound(self) -> float:
        """Get the current best lower bound from biqbin C solver

        Returns:
            float: global lower bound
        """
        return self.__biqbin.get_lower_bound()

    # unused currently
    def _set_lower_bound(self, new_lb: float):
        """Set lower bound manually in biqbin C solver.

        Args:
            new_lb (float): lower bound to set it to
        """
        self.__biqbin.set_lower_bound(new_lb)

    ###########################################################
    ###################### Bab FUNCTIONS  #####################
    ###########################################################

    def _new_node(self, parent_node: _BabNode = None) -> _BabNode:
        """Generates a new node from parent or root node if parent is None

        Args:
            parent_node (_BabNode): Node that we are branching from

        Returns:
            _BabNode: child node
        """
        parent = ctypes.pointer(parent_node) if parent_node else None
        return self.__biqbin.new_node(parent).contents

    ###########################################################
    ############# wrapped_bounding.c FUNCTIONS  ###############
    ###########################################################
    def _create_subproblem(self, babnode: _BabNode, globals: _GlobalVariables):
        """Must happen first in evaluation!
        Creates subproblem PP and sets it in globals struct

        Args:
            babnode (_BabNode): current node being evaluated
            globals (_GlobalVariables): global variables ctype.Structure, must be initialized before
        """
        self.__biqbin.create_subproblem(
            ctypes.pointer(babnode),
            globals.SP,
            globals.PP
        )

    def _init_sdp(self, node: _BabNode, sol_x, globals: _GlobalVariables):
        """Initialize SDP bound. Builds the temp solution x to be used in further evaluations

        Args:
            node (_BabNode): current node
            sol_x (_type_): stored solution for evaluation
            globals (_GlobalVariables): global variables
        """
        self.__biqbin.init_sdp(
            ctypes.pointer(node),
            sol_x,
            ctypes.pointer(globals)
        )

    def _init_main_sdp_loop(self, globals: _GlobalVariables, rank: int) -> bool:
        """_summary_

        Args:
            globals (_GlobalVariables): _description_
            rank (int): _description_

        Returns:
            bool: True if node can be pruned or given up on
        """
        is_root = 1 if rank == 0 else 0  # runs differently on root node
        return self.__biqbin.init_main_sdp_loop(
            ctypes.pointer(globals),
            is_root
        ) != 0

    def _main_sdp_loop_start(self, globals: _GlobalVariables) -> bool:
        """Before heuristics is used in sdp_bound loop.
        Uses bundle_method, calculates upper bound for this iteration of the loop.

        Args:
            globals (_GlobalVariables): global variables

        Returns:
            bool: True if node can be pruned
        """
        return self.__biqbin.main_sdp_loop_start(ctypes.pointer(globals))

    def _main_sdp_loop_end(self, node: _BabNode, globals: _GlobalVariables) -> bool:
        """Before heuristics is used in sdp_bound loop.
        Uses bundle_method, calculates upper bound for this iteration of the loop.

        Args:
            globals (_GlobalVariables): global variables

        Returns:
            bool: True if evaluating is over
        """
        return self.__biqbin.main_sdp_loop_end(
            ctypes.pointer(node),
            ctypes.pointer(globals)
        ) != 0

    def _set_globals_diff(self, globals):
        """If evaluating root node, globals->diff needs to be set

        Args:
            globals (_type_): global variables
        """
        self.__biqbin.set_globals_diff(ctypes.pointer(globals))

    def _get_upper_bound(self, node: _BabNode, globals: _GlobalVariables) -> float:
        """After evaluation is complete, get the calculated upper bound (globals->f + fixed_value)

        Args:
            node (_BabNode): current node
            globals (_GlobalVariables): global variables

        Returns:
            float: upper bound
        """
        return self.__biqbin.get_upper_bound(
            ctypes.pointer(node),
            ctypes.pointer(globals)
        )

    def _update_solution_wrapped(self, node: _BabNode, sol_x, globals: _GlobalVariables):
        """Updates lower bound and solution nodes based on the temp solution x, stores in node->sol.X if it is better

        Args:
            babnode (_BabNode): current node
            sol_x (_type_): solution found
            globals (_GlobalVariables): globals variables
        """
        self.__biqbin.update_solution_wrapped(
            ctypes.pointer(node),
            sol_x,
            globals.SP
        )
    ###########################################################
    ################# heuristic.c FUNCTIONS  ##################
    ###########################################################

    def _GW_heuristics(self, node: _BabNode, sol_x, globals: _GlobalVariables) -> float:
        """Goemans-Williamson random hyperplane heuristic located in heuristic.c

        Args:
            babnode (_BabNode): current node
            sol_x (ctypes.Array[ctypes.c_int]): current best solution, gets updated here if a better one is found

        Returns:
            float: best lower bound found
        """
        return self.__biqbin.GW_heuristic(
            globals.SP,
            globals.PP,
            ctypes.pointer(node),
            sol_x,
            self.num_vertices,  # num variable ??
            globals.Z
        )

    ###########################################################
    ############ wrapped_heuristics.c FUNCTIONS  ##############
    ###########################################################
    def _cholesky_factorization(self, state: _HeurState, globals: _GlobalVariables) -> int:
        """Mathematics

        Args:
            state (_HeurState): persistant variables structures for heuristics functions

        Returns:
            bool: returns True if Cholesky factorization was successful
        """
        info = self.__biqbin.cholesky_factorization(
            ctypes.pointer(state),
            globals.Z
        )
        return info == 0

    def _initialize_heuristics(self, node: _BabNode, globals: _GlobalVariables) -> _HeurState:
        """Creates _HeurState struct in C solver and returns it,

        Args:
            babnode (_Babnode): current node

        Returns:
            _HeurState: stores the persistant variables that need to be passed to heuristics functions
        """
        heur_state = self.__biqbin.heuristic_init(
            globals.SP,
            globals.PP,
            ctypes.pointer(node),
            globals.X,
            globals.Z
        )
        return heur_state.contents

    def _postprocess_heuristics(self,
                                state: _HeurState,
                                babnode: _BabNode,
                                sol_x,
                                globals: _GlobalVariables,
                                heur_value: float
                                ) -> bool:
        """Runs after GW_heuristic, process the solution found there

        Args:
            state (_HeurState): persistant variables shared between heuristic functions
            babnode (_BabNode): current node
            sol_x (ctypes.Array[ctypes.c_int]): best solution found for the given node and subproblem (globals->PP)
            heur_value (float): best lower bound found for the current node and subproblem (globals->PP)

        Returns:
            bool: True if GW_heuristic solution is better than before
        """
        success = self.__biqbin.heuristic_postprocess(
            ctypes.pointer(state),
            ctypes.pointer(babnode),
            sol_x,
            globals.X,
            globals.Z,
            heur_value
        )
        return success == 1

    def _finalize_heuristics(self, state: _HeurState) -> float:
        """Frees _HeurState memory in C

        Args:
            state (_HeurState): _description_

        Returns:
            float: state.fh (lower bound)
        """
        return self.__biqbin.heuristic_finalize(ctypes.pointer(state))

    ##############################################################
    ##################### Unit test functions ####################
    ##############################################################

    def set_random_seed(self, seed: int):
        self.__biqbin.srand(seed)

    def set_parameters(self, params: BiqBinParameters):
        self.params = params
        self.__biqbin.set_parameters(params.get_c_struct())

    def init_globals(self, L, num_verts: int) -> ctypes._Pointer:
        self.num_vertices = num_verts
        return self.__biqbin.init_globals(L, num_verts)

    def free_globals(self, globals_pointer):
        self.__biqbin.free_globals(globals_pointer)
