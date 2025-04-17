from typing import List, Optional
import ctypes
import numpy as np
from biqbin_data_objects import BiqBinParameters, _BabNode
from biqbin_base import _BiqbinBase


class ParallelBiqbin(_BiqbinBase):
    def __init__(self, params: BiqBinParameters = BiqBinParameters()):
        super().__init__()  # sets the calls to C functions
        self.params: BiqBinParameters = params
        self.rank: int = None  # MPI rank

        self.num_vertices: int = 0
        self.globals_p: ctypes._Pointer = None

    def compute(self, graph_path: str):
        """Computes the entire solution for the given graph instance.

        Args:
            graph_path (str): path to graph instance file.
        """
        # init MPI in C, get rank
        args = ["./biqbin.so",
                graph_path,
                ]
        self.rank = self._init_mpi(len(args), args)

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
                over = self._master_main_loop()
            # tell workers to end, release memory and finalize MPI
            self._master_end()

        else:
            # Initialize solver for workers, needs params, master lets them know if is over
            over = self.__worker_init()
            while not over:
                over = self.__worker_main_loop()
            # Free memory and finalize MPI
            self._worker_end()

    def compute_lower_bound(self, node: _BabNode, sol_x):
        # TODO run_heuristic finds a solution and it can be evaulated, need to figure out how it is connected to the rest of the evaluation
        pass

    def compute_upper_bound(self, _Babnode):
        # TODO run_heuristic is ran many times inside the upper bound evaluation
        pass

        #################### heuristics  ####################
    def run_heuristics(self, babnode: _BabNode, sol_x) -> float:
        """runs the entire heuristics, original runHeuristic in heuristic.c

        Args:
            babnode (_BabNode): current node
            sol_x : best solution for this node and subproblem (globals->PP)

        Returns:
            float: lower bound for the given solution
        """
        # heur_state stores the variables that are used in separate heuristic functions
        globals = self.globals_p.contents
        # initialized and get persistant variables accross heuristics functions
        heur_state = self._initialize_heuristics(babnode, globals)
        done = 0
        # do twice, reset by _postprocess_heuristics
        while done < 2:
            done += 1
            self._cholesky_factorization(heur_state, globals)
            # GW heuristic is unchanged
            heur_value = self._GW_heuristics(babnode, sol_x, globals)
            # reset if GW found a better solution than the one before
            if (self._postprocess_heuristics(heur_state, babnode, sol_x, globals, heur_value)):
                done = 0
        return self._finalize_heuristics(heur_state)

    ###########################################################
    ################### WORKER MAIN LOOP  #####################
    ###########################################################
    def __worker_main_loop(self) -> bool:
        """worker main loop in C, waits for the over signal, if not over receive babnode to process, branches and sends more nodes to other workers

        Returns:
            bool: over, True if done solving
        """
        # Wait for over signal
        over = self._worker_check_over()
        if over:
            return True
        # Get and update local best lower bound receive node through MPI, insert it into priority queue in C
        self._worker_receive_problem()

        # loops until worker becomes idle
        while not self._pq_is_empty():
            # check time limit
            if self.params.time_limit == 1 and self._time_limit_reached():
                return True

            # get node from PQ (heap.c)
            babnode = self._pq_pop()
            # Save previous g_lowerbound
            old_lb = self._get_lower_bound()
            # Evaluate Node
            self.__evaluate_node(babnode)
            # After eval, communicate new solution, frees node etc
            self._after_evaluation(babnode, old_lb)

        # if PQ empty notify master of being idle
        self._worker_send_idle()
        return False

    ###########################################################
    #################### Initialization  ######################
    ###########################################################
    def __master_init(self, filename: str, L: np.ndarray, num_verts: int, num_edge: int) -> bool:
        """Master initialization split into 3 stages, *before* and *after* root node gen/eval, and the root node generation and evaluation.

        Args:
            filename (str): graph_path for the creation of output file
            L (np.ndarray): Laplacian matrix created for SP->L
            num_verts (int): number of vertices in the graph
            num_edge (int): number of edges in the graph, needed only for the output print

        Returns:
            bool: True if finished solving (root node was pruned)
        """
        self._master_init_start(
            filename,
            L,
            num_verts,
            num_edge,
            self.params
        )
        self.globals_p = self._get_globals_pointer()

        # Root node generation and evaluation
        self._increase_num_eval_nodes()
        root_node = self._new_node()
        self.__evaluate_node(root_node)
        # After root node
        over = self._master_init_end(root_node)
        return over

    def __worker_init(self) -> bool:
        """workers first receive status if the solver is done, if not update the global lower bound

        Returns:
            bool: True if done solving
        """
        over = self._worker_init(self.params)
        self.globals_p = self._get_globals_pointer()
        self.num_vertices = self.globals_p.contents.SP.contents.n
        return over

    ##############################################################
    ##################### evaluate seperatered ###################
    ##############################################################
    def __evaluate_node(self, babnode: _BabNode):
        # testing to see if I need to store the pointer
        globals = self.globals_p.contents
        # changes globals->PP using globals->SP and current BabNode
        self._create_subproblem(babnode, globals)
        # Stores the best solution for node, updates it in BabNode x[BabPbSize] local variable
        sol_x = (ctypes.c_int * (self.num_vertices - 1))()
        # update solution_vector_x for the given subproblem and node
        self._init_sdp(babnode, sol_x, globals)
        # Run heuristics and update solution first time
        self.run_heuristics(babnode, sol_x)
        self._update_solution_wrapped(babnode, sol_x, globals)

        over = self._init_main_sdp_loop(
            self.globals_p.contents,
            self.rank
        )
        while not over:
            prune = self._main_sdp_loop_start(globals)
            if not prune:
                self.__update_partial_solution(babnode, sol_x)
                self.run_heuristics(babnode, sol_x)
                self._update_solution_wrapped(babnode, sol_x, globals)

            over = self._main_sdp_loop_end(babnode, globals)

        if self.rank == 0 and self.params.use_diff == 1:
            self._set_globals_diff(globals)
        return self._get_upper_bound(babnode, globals)

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

    ##############################################################
    ######################## Data parsing ########################
    ##############################################################

    def read_maxcut_input(self, graph_path: str) -> tuple[np.ndarray, int, int, str]:
        """
        Reads and parses the input graph file, default is in a weighted edge list,
        with the first line being number_of_vertices number_of_edges.

        Parameters:
            graph_path (str): path to file

        Returns:
            tuple: 
                - np.ndarray: The adjacency matrix.
                - int: Number of vertices.
                - int: Number of edges.
                - str: Name of the graph instance.
        """
        with open(graph_path, 'r') as f:
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

            name = graph_path
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

    def evaluate(self, node: _BabNode, globals, rank: int):
        self.globals_p = globals
        self.rank = rank
        return self.__evaluate_node(node)
