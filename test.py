from scipy.sparse import coo_matrix
import itertools
import numpy as np
import random
import ctypes
import json
import os

class BiqbinInput(ctypes.Structure):
    """ creates a struct to match emxArray_real_T """

    _fields_ = [
        ('m', ctypes.c_int), 
        ('n', ctypes.c_int), 
        ('A', ctypes.POINTER(ctypes.c_double)),
        ('b', ctypes.POINTER(ctypes.c_double)),
        ('F', ctypes.POINTER(ctypes.c_double)),
        ('c', ctypes.POINTER(ctypes.c_double)),
        ('name', ctypes.c_char * 255)
    ]

    
class BiqBinParameters(ctypes.Structure):
    """ creates a struct to match emxArray_real_T """

    _fields_ = [
        ('init_bundle_iter', ctypes.c_int), 
        ('max_bundle_iter', ctypes.c_int), 
        ('min_outer_iter', ctypes.c_int), 
        ('max_outer_iter', ctypes.c_int), 
        ('violated_Ineq', ctypes.c_double),
        ('TriIneq', ctypes.c_int), 
        ('Pent_Trials', ctypes.c_int), 
        ('Hepta_Trials', ctypes.c_int), 
        ('include_Pent', ctypes.c_int), 
        ('include_Hepta', ctypes.c_int), 
        ('root', ctypes.c_int), 
        ('use_diff', ctypes.c_int), 
        ('time_limi', ctypes.c_int), 
        ('branchingStrategy', ctypes.c_int), 
        ('detailedOutput', ctypes.c_int), 
    ]


def qbo_input_to_matrices(instance):

    # zes it is realy like this in biqbin :(
    def f(F):
        for i,j,v in F:
            if i==j:
                yield (i, j), v
            else:
                yield (i, j), v
                yield (j, i), v
                
    
    Fdict = dict(f(instance["F"]))
    Anp = np.array(instance["A"])
    cnp = np.array(instance["c"])
    bnp = np.array(instance["b"])

    Find, Fv = (list(Fdict.keys()), list(Fdict.values()))
    Find = np.asarray(Find)

    Fm = coo_matrix((Fv, (Find[:, 0], Find[:, 1])), shape=(instance["number_of_variables"], instance["number_of_variables"])).todense()
    Am = coo_matrix((Anp[:, 2], (Anp[:, 0], Anp[:, 1])), shape=(instance["number_of_constrains"], instance["number_of_variables"])).todense()
    cm = coo_matrix((cnp[:, 1], ([0]*len(instance["c"]), cnp[:, 0])), shape=(1, instance["number_of_variables"])).todense()
    bm = coo_matrix((bnp[:, 1], (bnp[:, 0], [0]*len(instance["b"]))), shape=(instance["number_of_constrains"], 1)).todense()
                   
                   
    instance["Fm"] = Fm
    instance["Am"] = Am
    instance["cm"] = cm
    instance["bm"] = bm
    
    return instance
 

with open("test/test-data.json", "r") as f:
    instance = json.load(f)
    
with open("test/params.json", "r") as f:
    params = json.load(f)

instance = qbo_input_to_matrices(instance)

biqbin_input = BiqbinInput(
    n = instance['number_of_variables'],
    m = instance['number_of_constrains'],
    A = instance['Am'].astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    b = instance['bm'].astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    F = instance['Fm'].astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    c = instance['cm'].astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    name = b'python-test'
)

biqbin_params = BiqBinParameters(**params)

biqbin = ctypes.CDLL(os.path.abspath("biqbin.so"))

biqbin.compute.argtypes = [BiqbinInput, BiqBinParameters]
biqbin.compute.restype = ctypes.c_int

biqbin.compute(biqbin_input, biqbin_params)
