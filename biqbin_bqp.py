import json
import scipy as sp
import numpy as np
from biqbin_base import DataGetterMaxCutDefault, BaseParser, MaxCutSolver

# these functions are placeholder implementations!
from bqp_data_processing_PLACEHOLDER import read_data_bqp, read_data_bqp_json, read_solution_bqp 

class ParserBQP(BaseParser):
    def __init__(self):
        super().__init__(prog=f'biqbin_bqp.py', description='Biqbin BQP solver')
        self.add_argument('-j', '--json', action='store_true', help='use json input file')
        
class DataGetterBQPDefault(DataGetterMaxCutDefault):
    """
    Uses the default C implementation of biqbin_general_bqp, reads and parses bqp instance file and parses
    into the adjacency matrix.
    """
    def read_file(self):
        print("PLACEHOLDER FUNCTION")
        self.adj_matrix = read_data_bqp(self.filename)
        return self.adj_matrix
    
class DataGetterBQPJson(DataGetterMaxCutDefault):
    """
    Uses the default json implementation of biqbin_general_bqp, reads and parses bqp instance file and parses
    into the adjacency matrix.
    """
    def read_file(self):
        print("PLACEHOLDER FUNCTION")
        instance = self.read_bqp_json(self.filename)
        self.adj_matrix = read_data_bqp_json(instance)
        return self.adj_matrix
    
    def read_bqp_json(self, filename):
        with open(filename, 'r') as f:
            instance = json.load(f)
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

        Fm = sp.sparse.coo_matrix((Fv, (Find[:, 0], Find[:, 1])), shape=(instance["number_of_variables"], instance["number_of_variables"])).todense()
        Am = sp.sparse.coo_matrix((Anp[:, 2], (Anp[:, 0], Anp[:, 1])), shape=(instance["number_of_constraints"], instance["number_of_variables"])).todense()
        cm = sp.sparse.coo_matrix((cnp[:, 1], ([0]*len(instance["c"]), cnp[:, 0])), shape=(1, instance["number_of_variables"])).todense()
        bm = sp.sparse.coo_matrix((bnp[:, 1], (bnp[:, 0], [0]*len(instance["b"]))), shape=(instance["number_of_constraints"], 1)).todense()
                    
                    
        instance["Fm"] = Fm
        instance["Am"] = Am
        instance["cm"] = cm
        instance["bm"] = bm
        
        return instance

if __name__ == '__main__':
    parser = ParserBQP()
    args = parser.parse_args()
    
    # Get the DataGetter for the BQP instance
    if args.json:
        data_getter = DataGetterBQPJson(args.problem_instance)
    else:
        data_getter = DataGetterBQPDefault(args.problem_instance)
        
        
    solver = MaxCutSolver(data_getter, args.params, args.time)
    result = solver.run()  # run the solver

    rank = solver.get_rank()
    if rank == 0:
        # Convert the Max-Cut solution back to BQP !! PLACEHOLDER FUNCTION !!
        result['bqp'] = read_solution_bqp(result, len(solver.data_getter.problem_instance()) - 1,)
        print(result)
        solver.save_result(result, output_path_in=args.output, overwrite=args.overwrite)
