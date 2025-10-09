import argparse
from biqbin_base import BQPSolver, DataGetterBQPDefault, ParserBQP

"""
    Default MaxCut Biqbin wrapper example
"""

if __name__ == '__main__':

    parser = ParserBQP()
    args = parser.parse_args()
    
    # Create an instance of the MaxCutSolver passing in the above arguments
    data_getter = DataGetterBQPDefault(args.problem_instance)

    solver = BQPSolver(data_getter, args.params, args.time)
    result = solver.run()  # run the solver

    rank = solver.get_rank()
    if rank == 0:
        # Print the results if master rank
        print(result)
        solver.save_result(result, output_path_in=args.output, overwrite=args.overwrite)
