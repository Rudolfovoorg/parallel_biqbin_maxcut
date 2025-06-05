import sys

from biqbin_base import QUBOSolver, DataGetterJson

"""
    Default Qubo solver using Biqbin MaxCut wrapper
"""

if __name__ == '__main__':
    # Path to qubo json file file and path to parameters file
    _, problem_instance_file_name, params = sys.argv

    # Instance of the default DataGetterJson class takes the path to qubo.json
    data_getter = DataGetterJson(problem_instance_file_name)
    # Initialize QUBOSolver class which takes a DataGetter class instance and path to parameters file
    solver = QUBOSolver(data_getter=data_getter, params=params)
    # Run biqbin solver
    result = solver.run()

    rank = solver.get_rank()
    if rank == 0:
        # Master rank prints the results
        print(result)
        solver.save_result(result)
