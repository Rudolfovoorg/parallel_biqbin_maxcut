import sys
from parallel_biqbin import ParallelBiqbin


graph_path = sys.argv[1]
params_path = sys.argv[2]

biqbin = ParallelBiqbin()

rank = biqbin.initialize(graph_path, params_path)

if rank == 0:
    over = biqbin.master_init()
    while not over:
        over = biqbin.master_main_loop()
    biqbin.master_end()

else:
    over = biqbin.worker_init()
    while not over:
        over = biqbin.worker_main_loop()
    biqbin.worker_end()
