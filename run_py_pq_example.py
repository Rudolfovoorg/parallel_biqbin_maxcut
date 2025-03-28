import sys
import time
import heapq
import numpy as np
from mpi4py import MPI
from parallel_biqbin_maxcut import ParallelBiqBinMaxCut, BabFunctions
from biqbin_data_objects import BabSolution
from helper_functions import HelperFunctions

"""
Priority queue and MPI communication done in python, needs further testing.
"""

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

biqbin = ParallelBiqBinMaxCut()
help = HelperFunctions()
num_vertices = 0
name = ''.encode('utf-8')
status = MPI.Status()
over = False
diff = None

if rank == 0:
    start_time = time.time()
    input_file_path = sys.argv[1]
    # Read Instance file
    adj_matrix, num_vertices, num_edges, name = help.read_maxcut_input(
        input_file_path)

    output_pre = f"Input file: {input_file_path}\n\nGraph has {num_vertices} vertices and {num_edges} edges."
    print(output_pre)
    # Construct L-matrix
    L_matrix = help.get_SP_L_matrix(adj_matrix)
    biqbin.open_output_file(name)

# Get shape of L from root process
num_vertices = comm.bcast(num_vertices, root=0)
# Allocate an empty array on non-root processes
if rank != 0:
    params_np_array = np.empty(21, dtype=np.float64)
    L_matrix = np.empty((num_vertices, num_vertices),
                        dtype=np.float64)  # Ensure the correct dtype

# Broadcast the L matrix
comm.Bcast(L_matrix, root=0)
# All of them read params from file
params = help.read_parameters_file(sys.argv[2])

# this allocates and sets SP and PP, srand, params, initBabsolution, globals->TIME
biqbin.init_solver(L_matrix, num_vertices, params)
babfuns = BabFunctions(L_matrix, num_vertices, params)


if rank == 0:
    free_workers = [i for i in range(1, size)]
    # Init_PQ() equivalent
    # create root node
    root_node = babfuns.generate_node()
    biqbin.evaluate(root_node, rank)
    # there is a small difference with this result between C and python
    root_node_upper_bound = root_node.upper_bound
    # Update solution, evaluate updates this in C
    sol_val = babfuns.evaluate_solution(root_node.sol)
    babfuns.best_lower_bound = sol_val
    babfuns.solution = root_node.sol

    if babfuns.best_lower_bound + 1 < root_node_upper_bound:
        heapq.heappush(babfuns.pq, (-root_node.upper_bound, root_node))

    # Broadcast over
    if len(babfuns.pq) == 0:
        over = True

    # broadcast diff
    diff = biqbin.get_diff()
    print(f"{diff = }")
    if params.use_diff > 0:
        diff = comm.bcast(diff, root=0)

    # this gets broadcaster every time anyway, but this way it's more 1 to 1
    over = comm.bcast(over, root=0)
    num_workers = 0
    while len(babfuns.pq) > 0:
        _, current_node = heapq.heappop(babfuns.pq)
        worker = free_workers.pop()
        babnode_np_array = babfuns.babnode_to_numpy(current_node)

        comm.send(over, worker, tag=0)
        comm.send(babfuns.best_lower_bound, worker, tag=1)
        comm.Send(babnode_np_array, worker, tag=2)
        num_workers += 1

    # Main loop of master if workers are busy
    while not over:
        over = len(free_workers) == size - 1
        if over:
            break
        value = comm.recv(source=MPI.ANY_SOURCE, status=status)
        tag = status.Get_tag()
        source = status.Get_source()
        """
        tags
        0 = over
        1 = idle
        2 = freeworkers
        3 = new_solution
        4 = babnode
        """
        if tag == 1:  # WORKER IDLE MESSAGE
            free_workers.append(source)

        elif tag == 2:  # REQUEST FREE WORKERS MESSAGE
            num_workers_to_send = min(value, len(free_workers))
            free_workers_to_send = []
            for i in range(num_workers_to_send):
                free_workers_to_send.append(free_workers.pop())

            comm.send(free_workers_to_send, dest=source)
            comm.send(babfuns.best_lower_bound, dest=source)
            num_workers += num_workers_to_send

        elif tag == 3:  # NEW VALUE MESSAGE
            new_lowerbound = value
            new_solution_array = np.zeros(babfuns.BabPbSize, dtype=np.int32)
            comm.Recv(new_solution_array, source=source)

            if new_lowerbound > babfuns.best_lower_bound:
                babfuns.best_lower_bound = new_lowerbound
                babfuns.solution = BabSolution(new_solution_array)
            comm.send(babfuns.best_lower_bound, dest=source)

    # print(f"{rank = } - {free_workers = }")
    for worker in free_workers:
        # print(f"sending over to {rank = }")
        comm.send(over, worker, tag=0)

# WORKER PROCESS MAIN LOOP
else:
    if params.use_diff > 0:
        diff = comm.bcast(diff, root=0)
        biqbin.set_diff(diff)  # set diff in C

    over = comm.bcast(over, root=0)
    while not over:
        # print(f"{rank = } WHILE NOT OVER")
        # First check if it is over
        over = comm.recv(source=MPI.ANY_SOURCE, tag=0)
        if over:
            break

        # Then update the lower bound
        babfuns.best_lower_bound = float(
            comm.recv(source=MPI.ANY_SOURCE, tag=1))
        biqbin.update_lowerbound(babfuns.best_lower_bound)
        # then get the node array
        node_array = np.zeros(3 * babfuns.BabPbSize + 2, dtype=np.float64)
        comm.Recv(node_array, source=MPI.ANY_SOURCE, tag=2)
        # Process node and decide how to move forward
        node = babfuns.numpy_to_babnode(node_array)
        heapq.heappush(babfuns.pq, (-node.upper_bound, node))
        while len(babfuns.pq) > 0:
            # print(f"{rank = } branching")
            _, current_node = heapq.heappop(babfuns.pq)
            current_best = babfuns.best_lower_bound  # save current best
            babfuns.branch(current_node, biqbin, rank)

            # update global best
            if (current_best > babfuns.best_lower_bound):
                comm.send(babfuns.best_lower_bound, dest=0)
                comm.Send(babfuns.solution.X, dest=0)
                babfuns.best_lower_bound = comm.recv(source=0)
                # print(f"{rank =} is updating lb to {babfuns.best_lower_bound}")

            # Request free workers
            if len(babfuns.pq) > 1:
                comm.send(len(babfuns.pq) - 1, dest=0, tag=2)
                received_workers = comm.recv(source=0)
                babfuns.best_lower_bound = comm.recv(source=0)

                for w in received_workers:
                    _, sent_node = heapq.heappop(babfuns.pq)
                    babnode_np_array = babfuns.babnode_to_numpy(sent_node)

                    comm.send(over, w, 0)
                    comm.send(babfuns.best_lower_bound, w, 1)
                    comm.Send(babnode_np_array, w, 2)
        comm.send(None, dest=0, tag=1)
        # print(f"{rank = } - over??")

# Use numpy array to store the sum
total_eval_nodes = np.zeros(1, dtype=np.int32)
if rank == 0:
    # Initialize a numpy array for total_eval_nodes to receive the sum at rank 0
    # Initialize num_eval_nodes for the root process
    num_eval_nodes = np.array([0], dtype=np.int32)  # Wrap in a numpy array
else:
    # For non-root processes, wrap the scalar in a numpy array
    num_eval_nodes = np.array([babfuns.num_eval_nodes - 1], dtype=np.int32)

# Perform the reduction across all processes to sum num_eval_nodes
comm.Reduce(num_eval_nodes, total_eval_nodes, op=MPI.SUM, root=0)

if rank == 0:
    end = time.time()
    babfuns.num_eval_nodes += total_eval_nodes[0]
    biqbin.close_output_file()
    # Final output string
    sol_str = babfuns.get_solution_string()
    output_post = f"\nNodes = {babfuns.num_eval_nodes}\nRoot node bound = {root_node_upper_bound:.2f}\nMaximum value = {babfuns.best_lower_bound:.0f}\n{sol_str}\nTime = {end - start_time:.2f} s\n"
    print(output_post)

    output_file_path = help.find_latest_output_file(
        input_file_path)  # output file is created by C
    if output_file_path:
        with open(output_file_path, "r") as f:
            output_mid = f.read()
        final_output_file_content = f"{output_pre}\n{output_mid}{output_post}"
        with open(output_file_path, "w") as f:
            f.write(final_output_file_content)
