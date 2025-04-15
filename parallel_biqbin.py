from typing import List
import ctypes
import numpy as np
from numpy.typing import NDArray
from biqbin_data_objects import BiqbinParameters, _BabNode, _HeurState
from biqbin_base import _BiqbinBase


class ParallelBiqbin(_BiqbinBase):
    def __init__(self, params: BiqbinParameters = BiqbinParameters()):
        super().__init__()  # sets the calls to C functions
        self.params = params
        self.rank = None  # MPI rank

        self.num_vertices: int = 0
        self._globals_p: ctypes.POINTER = None

    def compute(self, graph_path: str):
        """Computes the entire solution for the given graph instance.

        Args:
            graph_path (str): path to graph instance file.
        """
        # init MPI in C, get rank
        self.rank = self.__init_MPI(graph_path)

        if self.rank == 0:
            # Only rank 0 needs input data about the graph, name to open output file, L matrix and num verts for the problem
            adj, num_verts, num_edge, name = self.read_maxcut_input(
                graph_path)
            self.num_vertices = num_verts
            L_matrix = self.get_Laplacian_matrix(adj)
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
        """Get the current best lower bound from biqbin C solver

        Returns:
            float: global lower bound
        """
        return self._biqbin.get_lower_bound()

    def set_lower_bound(self, new_lb: float):
        """Set lower bound manually in biqbin C solver.

        Args:
            new_lb (float): lower bound to set it to
        """
        self._biqbin.set_lower_bound(new_lb)

    def time_limit_reached(self) -> bool:
        return self._biqbin.time_limit_reached() == 1

    def pop_node(self) -> _BabNode:
        """Pop node from the priority queue in heap.c

        Returns:
            _BabNode(ctype.Structure): defined in biqbin_data_objects.py
        """
        return self._biqbin.pq_pop().contents

    def pq_is_empty(self) -> bool:
        """Nodes get evaluated in while priority queue is not empty
        Returns:
            bool: True if PQ in heap.c is not empty
        """
        return self._biqbin.pq_is_empty() != 0

    # worker main loop in C, waits for either the over signal or babnode to process, branches and sends more nodes to other workers
    def __worker_main_loop(self) -> bool:
        # Wait for over signal
        over = self._biqbin.worker_check_over() != 0
        if over:
            return True
        # receive problem through MPI, insert it into priority queue in C
        self._biqbin.worker_receive_problem()

        # loops until worker becomes idle
        while not self.pq_is_empty():
            # check time limit
            if self.params.time_limit == 1 and self.time_limit_reached():
                return True

            # get node from PQ (heap.c)
            babnode = self.pop_node()
            # Save previous g_lowerbound
            old_lb = self.get_lower_bound()
            # Evaluate Node
            self.__evaluate_node(babnode)
            # After eval, communicate new solution, frees node etc
            self._biqbin.after_evaluation(ctypes.pointer(babnode), old_lb)

        # if PQ empty notify master of being idle
        self._biqbin.worker_send_idle()
        return False

    def __init_MPI(self, graph_path) -> int:
        """
        Initializes MPI in biqbin C solver

        Parameters:
            graph_path (str): path string to graph file

        Returns:
            int: MPI rank of process. 
        """
        args = [b"./biqbin",
                graph_path.encode("utf-8"),
                ]
        argv = (ctypes.c_char_p * len(args))(*args)
        rank = self._biqbin.initMPI(2, argv)
        return rank

    # master rank evaluates the root node and decides if further branching is needed
    def __master_init(self, filename: str, L: np.ndarray, num_verts: int, num_edge: int) -> bool:
        # returns 0 if not over
        self._biqbin.master_init(
            filename,
            L,
            num_verts,
            num_edge,
            self.params.get_c_struct()
        ) != 0
        self._globals_p = self._biqbin.get_globals_pointer()

        self._biqbin.increase_num_eval_nodes()
        root_node = self._biqbin.new_node(None)
        self.__evaluate_node(root_node.contents)
        over = self._biqbin.master_init_end(root_node)
        return over

    def __master_main_loop(self) -> bool:
        """Main loop for master rank, waits for communication from workers and responds until all are free

        Returns:
            bool: True if solving is over
        """
        return self._biqbin.master_main_loop() != 0

    # Sends over signal to workers, frees memory in C, print to and close output file and finalize MPI
    def __master_end(self):
        self._biqbin.master_end()
        self._biqbin.MPI_Finalize()

    # workers first receive status if the solver is done, if not update the global lower bound
    def __worker_init(self) -> bool:
        over = self._biqbin.worker_init(self.params.get_c_struct()) != 0
        self._globals_p = self._biqbin.get_globals_pointer()
        self.num_vertices = self._globals_p.contents.SP.contents.n - 1
        return over

    # Frees memory in worker process, finalizes MPI
    def __worker_end(self):
        self._biqbin.worker_end()
        self._biqbin.MPI_Finalize()

    ##############################################################
    ##################### evaluate seperatered ###################
    ##############################################################
    def __evaluate_node(self, babnode: _BabNode):
        # changes globals->PP using globals->SP and current BabNode
        self.__create_subproblem(babnode)
        # Stores the best solution for node, updates it in BabNode x[BabPbSize] local variable in SDPbound function
        sol_x = (ctypes.c_int * (self.num_vertices - 1))()
        # update solution_vector_x for the given subproblem and node
        self._biqbin.init_sdp(babnode, sol_x, self._globals_p)
        # Run heuristics and update solution first time
        self.run_heuristics(babnode, sol_x)
        self.__update_solution_wrapped(babnode, sol_x)

        is_root = 1 if self.rank == 0 else 0  # runs differently on root node
        over = self._biqbin.init_main_sdp_loop(
            self._globals_p,
            is_root
        )
        while not over:
            prune = self._biqbin.main_sdp_loop_start(self._globals_p)
            if not prune:
                self.__update_partial_solution(babnode, sol_x)
                self.run_heuristics(babnode, sol_x)
                self.__update_solution_wrapped(babnode, sol_x)

            over = self._biqbin.main_sdp_loop_end(
                babnode, self._globals_p
            )

        if self.rank == 0 and self.params.use_diff == 1:
            self._biqbin.set_globals_diff(self._globals_p)
        return self._biqbin.get_upper_bound(babnode, self._globals_p)

    def __update_partial_solution(self, babnode: _BabNode, sol_x: ctypes.Array[ctypes.c_int]):
        """copies node.sol.X into sol_x while node.xfixed is 1, else 0

        Args:
            babnode (_BabNode): current node
            sol_x (ctypes.Array[ctypes.c_int]): solution nodes
        """
        for i in range(self.num_vertices - 1):
            if babnode.xfixed[i]:
                sol_x[i] = babnode.sol.X[i]
            else:
                sol_x[i] = 0

    def __create_subproblem(self, babnode: _BabNode):
        """Creates subproblem PP and sets it in globals struct

        Args:
            babnode (_BabNode): current node being evaluated
        """
        main_problem_SP = self._globals_p.contents.SP
        subproblem_PP = self._globals_p.contents.PP
        self._biqbin.create_subproblem(
            ctypes.pointer(babnode),
            main_problem_SP,
            subproblem_PP
        )

    ##############################################################
    #################### heuristics separated ####################
    ##############################################################
    def run_heuristics(self, babnode: _BabNode, sol_x: ctypes.Array[ctypes.c_int]) -> float:
        """runs the entire heuristics, original runHeuristic in heuristic.c

        Args:
            babnode (_BabNode): current node
            sol_x (ctypes.Array[ctypes.c_int]): best solution for this node and subproblem (globals->PP)

        Returns:
            float: lower bound for the given solution
        """
        # heur_state stores the variables that are used in separate heuristic functions
        heur_state = self.initialize_heuristics(babnode)
        done = 0
        while done < 2:
            done += 1
            self.cholesky_factorization(heur_state)
            heur_value = self.run_GW_heuristics(babnode, sol_x)
            if (self.postprocess_heuristics(heur_state, babnode, sol_x, heur_value)):
                done = 0
        return self.finalize_heuristics(heur_state)

    def run_GW_heuristics(self, babnode: _BabNode, sol_x: ctypes.Array[ctypes.c_int]) -> float:
        """Goemans-Williamson random hyperplane heuristic located in heuristic.c

        Args:
            babnode (_BabNode): current node
            sol_x (ctypes.Array[ctypes.c_int]): current best solution, gets updated here if a better one is found

        Returns:
            float: best lower bound found
        """
        globals = self._globals_p.contents
        return self._biqbin.GW_heuristic(
            globals.SP,
            globals.PP,
            ctypes.pointer(babnode),
            sol_x,
            self.num_vertices,  # num variable ??
            globals.Z
        )

    def initialize_heuristics(self, babnode: _BabNode) -> _HeurState:
        """Creates _HeurState struct in C solver and returns it,

        Args:
            babnode (_Babnode): current node

        Returns:
            _HeurState: stores the persistant variables that need to be passed to heuristics functions
        """
        globals = self._globals_p.contents
        heur_state = self._biqbin.heuristic_init(
            globals.SP,
            globals.PP,
            ctypes.pointer(babnode),
            globals.X,
            globals.Z
        )
        return heur_state.contents

    def cholesky_factorization(self, state: _HeurState) -> int:
        """Mathematics

        Args:
            state (_HeurState): persistant variables structures for heuristics functions

        Returns:
            bool: returns True if Cholesky factorization did not fail
        """
        info = self._biqbin.cholesky_factorization(
            ctypes.pointer(state),
            self._globals_p.contents.Z
        )
        return info == 0

    def postprocess_heuristics(self, state: _HeurState, babnode: _BabNode, sol_x: ctypes.Array[ctypes.c_int], heur_value: float) -> bool:
        """Runs after GW_heuristic, process the solution found there

        Args:
            state (_HeurState): persistant variables shared between heuristic functions
            babnode (_BabNode): current node
            sol_x (ctypes.Array[ctypes.c_int]): best solution found for the given node and subproblem (globals->PP)
            heur_value (float): best lower bound found for the current node and subproblem (globals->PP)

        Returns:
            bool: True if GW_heuristic solution is better than before
        """
        globals = self._globals_p.contents
        success = self._biqbin.heuristic_postprocess(
            ctypes.pointer(state),
            ctypes.pointer(babnode),
            sol_x,
            globals.X,
            globals.Z,
            heur_value
        )
        return success == 1

    def finalize_heuristics(self, state: _HeurState) -> float:
        """Frees _HeurState memory in C, 

        Args:
            state (_HeurState): _description_

        Returns:
            float: _description_
        """
        return self._biqbin.heuristic_finalize(ctypes.pointer(state))

    def __update_solution_wrapped(self, babnode: _BabNode, sol_x: ctypes.Array[ctypes.c_int]):
        # if the heuristic solution (lower bound) in *x is better than the old one, it will update it in C
        self._biqbin.update_solution_wrapped(
            babnode,
            sol_x,
            self._globals_p.contents.SP
        )

    ##############################################################
    ######################## Data parsing ########################
    ##############################################################
    def read_maxcut_input(self, filename) -> tuple[np.ndarray, int, int, str]:
        """
        Reads and parses the input graph file, default is in a weighted edge list,
        with the first line being number_of_vertices number_of_edges.

        Parameters:
            filename (str): path to file

        Returns:
            tuple: 
                - np.ndarray: The adjacency matrix.
                - int: Number of vertices.
                - int: Number of edges.
                - str: Name of the graph instance (default is filename.encode('utf-8')).
        """
        with open(filename, 'r') as f:
            # Read number of vertices and edges
            num_vertices, num_edges = map(int, f.readline().split())

            # Allocate adjacency matrix as a contiguous array (row-major)
            adj_matrix = np.zeros(
                (num_vertices, num_vertices), dtype=np.float64)

            # Read edges
            for _ in range(num_edges):
                i, j, weight = f.readline().split()
                i, j = int(i) - 1, int(j) - 1
                weight = float(weight)

                adj_matrix[i, j] = weight
                adj_matrix[j, i] = weight

            name = filename.encode('utf-8')
            return adj_matrix, num_vertices, num_edges, name

    def get_Laplacian_matrix(self, Adj: np.ndarray):
        """
        Constructs the extended Laplacian matrix.

        The resulting matrix L is structured as:
            [ L0         L0 @ e ]
            [ (L0 @ e)^T  e^T @ L0 @ e ]
        Where L0 is the standard graph Laplacian (degree matrix - adjacency matrix),
        and `e` is the all-ones vector.

        Parameters:
            Adj (np.ndarray): The adjacency matrix of the graph (shape: [n, n]).

        Returns:
            np.ndarray: A (n x n) matrix representing the extended Laplacian.
        """

        num_vertices = Adj.shape[0]

        # Initialize Laplacian matrix L
        L = np.zeros((num_vertices, num_vertices))

        # Compute Adje = Adj * e (sum of each row)
        Adje = Adj.sum(axis=1)

        # Compute Diag(Adje) - diagonal matrix with Adje values
        tmp = np.diag(Adje)

        # Construct Laplacian matrix: L = tmp - Adj, excluding the last row and column
        L[:-1, :-1] = tmp[:-1, :-1] - Adj[:-1, :-1]

        # Compute vector parts and constant part
        sum_row = L[:-1, :-1].sum(axis=1)
        sum_total = sum_row.sum()

        # Fill vector parts and constant term
        L[:-1, -1] = sum_row
        L[-1, :-1] = sum_row
        L[-1, -1] = sum_total

        return L

    ##############################################################
    ##################### Unit test functions ####################
    ##############################################################

    def set_random_seed(self, seed: int):
        self._biqbin.srand(seed)

    def set_parameters(self, params: BiqbinParameters):
        self.params = params
        self._biqbin.setParams(params.get_c_struct())

    def get_globals(self, L: NDArray[np.float64], num_verts: int):
        self.num_vertices = num_verts
        return self._biqbin.init_globals(L, num_verts)

    def free_globals(self, globals):
        self._biqbin.free_globals(globals)

    def evaluate(self, node: _BabNode, globals, rank: int):
        self._globals_p = globals
        self.rank = rank
        return self.__evaluate_node(node)
