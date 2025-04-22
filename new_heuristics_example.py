import sys
from parallel_biqbin import ParallelBiqbin
from biqbin_data_objects import BiqBinParameters, heuristicmethod
from random import randint

# path to the graphs file (i.e. "Instances/rudy/g05.60.0" or "test/Instances/rudy/g05_100.3")
graph_path = sys.argv[1]
# path to parameters (i.e. "test/params" or "params")
params_path = sys.argv[2]

# create a parameters instance to set non-default parameters, file_path is optional
parameters = BiqBinParameters(params_filepath=params_path)


class ExampleHeuristic(ParallelBiqbin):
    # wrape the heuristic() with @heuristicmethod located in biqbin_data_objects.py
    @heuristicmethod
    def heuristic(self, node, solution_out, globals):
        """_summary_

        Args:
            node (_BabNode): Current node ctypes.Structure class, contains information about the node
            solution_out (np.ndarray): 0, 1 array for included/excluded nodes in the solution, evaluated after this function to get the lower bound
            globals (_GlobalVariables): global variables class used by the C solver
        """
        # Mock example of a heuristic function
        # For each node in the solution
        for i in range(solution_out.shape[0]):
            # pass if node not fixed
            if node.xfixed[i] == 0:
                # asign 0 or 1 randomly
                solution_out[i] = randint(0, 1)


biqbin = ExampleHeuristic(params=parameters)
biqbin.compute(graph_path=graph_path)  # Compute runs the solver
