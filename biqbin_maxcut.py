import sys
from biqbin_base import MaxCutSolver

"""
    Default MaxCut Biqbin wrapper example
"""

if __name__ == '__main__':
    # needs path to graph file and path to parameters file
    _, problem_instance_file_name, params = sys.argv

    # Create an instance of the MaxCutSolver passing in the above arguments
    solver = MaxCutSolver(problem_instance_file_name, params)
    result = solver.run()  # run the solver

    rank = solver.get_rank()
    if rank == 0:
        # Print the results if master rank
        print(result)
        solver.save_result(result)
