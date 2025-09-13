import argparse
from biqbin_base import MaxCutSolver, DataGetterMaxCutDefault, DataGetterAdjacencyJson, ParserMaxCut

"""
    Default MaxCut Biqbin wrapper example
"""

if __name__ == '__main__':

    parser = ParserMaxCut()
    args = parser.parse_args()
    
    # Create an instance of the MaxCutSolver passing in the above arguments
    if args.edge_weight:
        data_getter = DataGetterMaxCutDefault(args.problem_instance)
    else:
        data_getter = DataGetterAdjacencyJson(args.problem_instance)
    solver = MaxCutSolver(data_getter, args.params, args.time)
    result = solver.run()  # run the solver

    rank = solver.get_rank()
    if rank == 0:
        # Print the results if master rank
        print(result)
        solver.save_result(result, output_path_in=args.output, overwrite=args.overwrite)
