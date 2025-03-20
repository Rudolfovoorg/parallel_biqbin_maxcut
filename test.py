import sys
from parallel_biqbin import ParallelBiqbin
from helper_functions import HelperFunctions

biqbin = ParallelBiqbin()
help = HelperFunctions()

graph_path = sys.argv[1]
params_path = sys.argv[2]

# init MPI in C, get rank
rank = biqbin.init_MPI(graph_path, params_path)

# everyone reads params file
params = help.read_parameters_file(params_path)

if rank == 0:
    # Only rank 0 needs input data about the graph, name to open output file, L matrix and num verts for the problem
    adj, num_verts, num_edge, name = help.read_maxcut_input(graph_path)
    L_matrix = help.get_SP_L_matrix(adj)
    # initialize master, if over == True don't go into main loop
    over = biqbin.master_init(name, L_matrix, num_verts, num_edge, params)
    while not over:
        over = biqbin.master_main_loop()
    # tell workers to end, release memory and finalize MPI
    biqbin.master_end()

else:
    # Initialize solver for workers, needs params, master lets them know if is over
    over = biqbin.worker_init(params)
    while not over:
        over = biqbin.worker_main_loop()
    # Free memory and finalize MPI
    biqbin.worker_end()
