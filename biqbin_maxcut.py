import argparse
from biqbin_base import MaxCutSolver, MaxCutFromJson, MaxCutFromEdgeWeights, ParserMaxCut, MaxCutToJson

"""
    Default MaxCut Biqbin wrapper example
"""

if __name__ == '__main__':

    parser = ParserMaxCut()
    args = parser.parse_args()
    
    # Create an instance of the MaxCutSolver passing in the above arguments
    if args.edge_weight:
        file_loader = MaxCutFromEdgeWeights(args.problem_instance)
    else:
        file_loader = MaxCutFromJson(args.problem_instance)
    
    problem = file_loader.read()
    solver = MaxCutSolver(problem, args.params, args.time)
    solution = solver.run()

    rank = solver.get_rank()
    if rank == 0:
        # Print the results if master rank
        if solution is None:
            raise ValueError(f'Solution to problem {problem} not found!')
        print(solution)
        MaxCutToJson(args.problem_instance + '.output.json', args.overwrite).write(solution)
