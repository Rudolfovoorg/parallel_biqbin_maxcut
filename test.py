import ctypes
import os
import sys

biqbin = ctypes.CDLL(os.path.abspath("biqbin.so"))
biqbin.wrapped_main.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]

biqbin.initMPI.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]

biqbin.initSolver.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
biqbin.initSolver.restype = ctypes.c_int

biqbin.getRank.restype = int

path = sys.argv[1]
param_path = sys.argv[2]
args = [b"./biqbin", path.encode("utf-8"), param_path.encode("utf-8")]
argc = len(args)

# Create an array of c_char_p (char pointers)
argv = (ctypes.c_char_p * argc)(*args)

biqbin.initMPI(argc, argv)
success = biqbin.initSolver(argc, argv) == 0

if not success:
    raise Exception("Data not parsed correctly")

rank = biqbin.getRank()
biqbin.compute()
