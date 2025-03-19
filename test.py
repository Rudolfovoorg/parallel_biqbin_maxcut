import ctypes
import os
import sys

biqbin = ctypes.CDLL(os.path.abspath("biqbin.so"))
biqbin.wrapped_main.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]

biqbin.initMPI.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
biqbin.initMPI.restype = int

biqbin.initSolver.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
biqbin.initSolver.restype = int

biqbin.master_init.restype = int
biqbin.master_main_loop.restype = int

biqbin.worker_init.restype = int

biqbin.worker_main_loop.restype = int

biqbin.getRank.restype = int

path = sys.argv[1]
param_path = sys.argv[2]
args = [b"./biqbin", path.encode("utf-8"), param_path.encode("utf-8")]
argc = len(args)

# Create an array of c_char_p (char pointers)
argv = (ctypes.c_char_p * argc)(*args)

rank = biqbin.initMPI(argc, argv)
success = biqbin.initSolver(argc, argv) == 0

if not success:
    raise Exception("Data not parsed correctly")

# if rank == 0:
#     biqbin.master_compute()

# else:
#     biqbin.worker_compute()


if rank == 0:
    over = biqbin.master_init()
    while over == 0:
        over = biqbin.master_main_loop()
    biqbin.master_end()

else:
    over = biqbin.worker_init()
    while over == 0:
        over = biqbin.worker_main_loop()
    biqbin.worker_end()

biqbin.finalizeMPI()
