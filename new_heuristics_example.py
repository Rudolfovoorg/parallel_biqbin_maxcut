import sys
from parallel_biqbin import ParallelBiqbin, heuristicmethod
from biqbin_data_objects import BiqBinParameters
import numpy as np

# path to the graphs file (i.e. "Instances/rudy/g05.60.0" or "test/Instances/rudy/g05_100.3")
graph_path = sys.argv[1]
# path to parameters (i.e. "test/params" or "params")
params_path = sys.argv[2]

# create a parameters instance to set non-default parameters, file_path is optional
parameters = BiqBinParameters(params_filepath=params_path)
# Read from params file if needed, can be set in the object


class ExampleHeuristic(ParallelBiqbin):
    @heuristicmethod
    def heuristic(self, node, solution_out, globals):
        heur_b = self.evaluate_solution(solution_out)
        return heur_b * 2


biqbin = ExampleHeuristic(params=parameters)
biqbin.compute(graph_path=graph_path)  # Compute runs the solver
