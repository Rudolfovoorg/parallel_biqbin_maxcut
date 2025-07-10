import sys

from biqbin_base import QUBOSolver, DataGetterJson, ParserQubo

"""
    Default Qubo solver using Biqbin MaxCut wrapper
"""

if __name__ == '__main__':
    # Path to qubo json file file and path to parameters file
    parser = ParserQubo()
    args = parser.parse_args()
    # Instance of the default DataGetterJson class takes the path to qubo.json
    data_getter = DataGetterJson(args.problem_instance)
    # Initialize QUBOSolver class which takes a DataGetter class instance and path to parameters file
    solver = QUBOSolver(data_getter=data_getter, params=args.params)
    # Run biqbin solver
    result = solver.run()

    rank = solver.get_rank()
    if rank == 0:
        # Master rank prints the results
        print(result)
        solver.save_result(result, args.output, args.overwrite)
