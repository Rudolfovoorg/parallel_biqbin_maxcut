import os
import glob
import re
import time
import heapq
import ctypes
from typing import Optional
import numpy as np
from biqbin_data_objects import BiqBinParameters, BabSolution, BabNode


class ParallelBiqBinMaxCut:
    def __init__(self, biqbin_path="biqbin.so"):
        # Load the shared library
        if not os.path.exists(biqbin_path):
            raise FileNotFoundError(f"Shared library not found: {biqbin_path}")
        self.biqbin = ctypes.CDLL(os.path.abspath(biqbin_path))

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

        self.biqbin.InitSolverWrapped.restype = None

        # Update lower bound
        self.biqbin.LBUpdate.argtypes = [ctypes.c_double]
        self.biqbin.LBUpdate.restype = None

        # Evaluate Node
        self.biqbin.EvaluateWrapped.argtypes = [
            ctypes.POINTER(BabNode),
            ctypes.c_int
        ]
        self.biqbin.EvaluateWrapped.restype = None

        # Get diff from Master
        self.biqbin.getDiff.argtypes = None
        self.biqbin.getDiff.restype = ctypes.c_double
        # Set diff at workers
        self.biqbin.setDiff.argtypes = [ctypes.c_double]
        self.biqbin.setDiff.restype = None

        # Read parameters
        self.biqbin.readParameters.argtypes = [ctypes.c_char_p]
        self.biqbin.readParameters.restype = BiqBinParameters

        # Open output file
        self.biqbin.openOutputFile.argtypes = [ctypes.c_char_p]
        self.biqbin.openOutputFile.restype = None

        # Free memory, reset solver
        self.biqbin.freeMemory.argtypes = None
        self.biqbin.freeMemory.restype = None
        # self.biqbin.reset_branch_and_bound_globals.argtypes = None
        # self.biqbin.reset_branch_and_bound_globals.restype = None

        # Close output file
        self.biqbin.closeOutputFile.argtypes = None
        self.biqbin.closeOutputFile.restype = None

    def init_solver(self, L_matrix, num_vertices, params):
        self.biqbin.InitSolverWrapped(L_matrix, num_vertices, params)

    def open_output_file(self, name):
        self.biqbin.openOutputFile(name)

    def close_output_file(self):
        self.biqbin.closeOutputFile()

    def evaluate(self, node: BabNode, rank: int):
        self.biqbin.EvaluateWrapped(ctypes.pointer(node), rank)

    def get_diff(self):
        return self.biqbin.getDiff()

    def set_diff(self, diff: float):
        self.biqbin.setDiff(diff)

    def update_lowerbound(self, new_lb: float):
        self.biqbin.LBUpdate(new_lb)

    def reset_solver(self):
        self.biqbin.reset_branch_and_bound_globals()
        self.biqbin.freeMemory()


# Branch and bound helper functions
class BabFunctions:
    def __init__(self, L_mat, num_vert: int, params: BiqBinParameters):
        self.L_mat = L_mat
        self.BabPbSize = num_vert - 1
        self.root_bound = None

        self.branching_strategy = params.branchingStrategy

        self.solution: Optional[BabSolution] = None
        self.best_lower_bound: Optional[float] = 0

        self.num_eval_nodes = 1

        self.pq = []

    def generate_node(self, parent_node: Optional[BabNode] = None):
        node = BabNode()

        for i in range(self.BabPbSize):  # Loop over all possible indices
            if parent_node is None:
                node.xfixed[i] = 0
                node.sol.X[i] = 0
            else:
                node.xfixed[i] = parent_node.xfixed[i]
                node.sol.X[i] = parent_node.sol.X[i] if node.xfixed[i] else 0

        # Set level: root node starts at 0, children are +1 from parent
        node.level = 0 if parent_node is None else parent_node.level + 1
        return node

    # Convert BabNode c_type object to numpy array for MPI transit
    def babnode_to_numpy(self, node):
        arr = np.zeros(3 * self.BabPbSize + 2, dtype=np.float64)
        arr[:self.BabPbSize] = np.frombuffer(
            node.xfixed, dtype=np.int32, count=self.BabPbSize).astype(np.float64)
        arr[self.BabPbSize:2 * self.BabPbSize] = np.frombuffer(
            node.sol.X, dtype=np.int32, count=self.BabPbSize).astype(np.float64)
        arr[2*self.BabPbSize:3 * self.BabPbSize] = np.frombuffer(
            node.fracsol, dtype=np.float64, count=self.BabPbSize)
        arr[3*self.BabPbSize] = float(node.level)
        arr[3*self.BabPbSize+1] = node.upper_bound
        return arr

    # Convert the numpy array back into BabNode to be able to pass it into the solver
    def numpy_to_babnode(self, arr):
        node = BabNode()
        # Only update first NMAX elements; leave rest of the full-sized buffer untouched
        node.xfixed[:self.BabPbSize] = arr[:self.BabPbSize].astype(np.int32)
        node.sol.X[:self.BabPbSize] = arr[self.BabPbSize:2 *
                                          self.BabPbSize].astype(np.int32)
        node.fracsol[:self.BabPbSize] = arr[2*self.BabPbSize:3*self.BabPbSize]

        node.level = int(arr[3*self.BabPbSize])
        node.upper_bound = arr[3*self.BabPbSize + 1]

        return node

    def get_branching_variable(self, node: BabNode):
        # Convert ctypes arrays to NumPy arrays
        xfixed = np.frombuffer(node.xfixed, dtype=np.int32)[:self.BabPbSize]
        fracsol = np.frombuffer(node.fracsol, dtype=np.float64)[
            :self.BabPbSize]

        # Create a boolean mask for unfixed variables (True = not fixed, False = fixed)
        unfixed_mask = xfixed == 0
        fractional_diffs = np.abs(0.5 - fracsol)

        if self.branching_strategy == 0:  # Least fractional
            # Mask values for fixed variables
            ic = np.argmax(fractional_diffs * unfixed_mask)
        elif self.branching_strategy == 1:  # Most fractional
            # Use np.where() safely
            ic = np.argmin(np.where(unfixed_mask, fractional_diffs, np.inf))
        else:
            raise ValueError("Error: Wrong value for branching_strategy")

        # Ensure only valid indices are returned
        return ic if unfixed_mask[ic] else -1

    def count_fixed_variables(self, node: BabNode) -> int:
        """
        Counts the number of fixed variables in the BabNode.

        Parameters:
        - node: BabNode object

        Returns:
        - The number of fixed variables
        """
        xfixed = np.frombuffer(node.xfixed, dtype=np.int32)[:self.BabPbSize]
        return np.count_nonzero(xfixed)

    def is_leaf_node(self, node: BabNode) -> bool:
        return self.count_fixed_variables(node) == self.BabPbSize

    # branches of a node, calls evaluate for both children nodes using C biqbin,
    # if needed pushes them into the priority queue
    def branch(self, node: BabNode, biqbin: ParallelBiqBinMaxCut, rank: int):
        biqbin.evaluate(node, rank)

        if not (self.best_lower_bound + 1 < node.upper_bound):
            return

        ic = self.get_branching_variable(node)

        for xic in range(2):  # 0 - 1
            child_node = self.generate_node(node)
            child_node.xfixed[ic] = 1
            child_node.sol.X[ic] = xic
            self.num_eval_nodes += 1
            if (self.is_leaf_node(child_node)):
                sol_val = self.evaluate_solution(child_node.sol)
                if (sol_val > self.best_lower_bound):
                    self.solution = child_node.sol
                    self.best_lower_bound = sol_val
                    print(
                        f"Node {self.num_eval_nodes} Feasible solution {sol_val:2}")
            else:
                # biqbin.evaluate(child_node, rank)
                heapq.heappush(self.pq, (-child_node.upper_bound, child_node))

    def evaluate_solution(self, sol: BabSolution) -> float:
        sol_val = 0
        for i in range(self.BabPbSize):
            for j in range(self.BabPbSize):
                sol_val += self.L_mat[i][j] * sol.X[i] * sol.X[j]
        return sol_val

    def get_solution_string(self):
        sol_str = "Solution = ("
        for i in range(self.BabPbSize):
            if self.solution.X[i] == 1:
                sol_str += f" {i + 1}"
        sol_str += " )"
        return sol_str
