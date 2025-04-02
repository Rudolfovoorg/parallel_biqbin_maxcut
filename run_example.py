import sys
from parallel_biqbin import ParallelBiqbin
from helper_functions import HelperFunctions
from biqbin_data_objects import ParametersWrapper

# path to the graphs file (i.e. "Instances/rudy/g05.60.0" or "test/Instances/rudy/g05_100.3")
graph_path = sys.argv[1]
# path to parameters (i.e. "test/params" or "params") optional
params_path = sys.argv[2]

parameters = ParametersWrapper()
# Read from params file if needed, can be set in the object
parameters.read_from_file(params_path)

biqbin = ParallelBiqbin(params=parameters)  # Biqbin Wrapper, runs C functions
# biqbin = ParallelBiqbin() -- can be run with default parameters
biqbin.compute(graph_path=graph_path)  # Compute runs the solver
