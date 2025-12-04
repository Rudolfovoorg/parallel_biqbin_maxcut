import argparse
from biqbin_base import MaxCutSolver, MaxCutFromJson, MaxCutFromEdgeWeights, ParserMaxCut, MaxCutToJson

"""
    Default MaxCut Biqbin wrapper example
"""

if __name__ == '__main__':

    parser = ParserMaxCut()
    args = parser.parse_args()

    # Select the file reader based on the file format
    if args.edge_weight:
        file_reader = MaxCutFromEdgeWeights()
    else:
        file_reader = MaxCutFromJson()

    # Read the file
    problem = file_reader.read(
        args.problem_instance, optimize_input=args.optimize)
    
    problem.verbose = args.verbose
    # Create an instance of the MaxCutSolver passing in path to params file and time limit
    solver = MaxCutSolver(args.params, args.time)
    # Compute the solution for the given problem
    solution = solver.compute(problem=problem)
    # Get rank to only save the results on the master rank
    rank = solver.get_rank()
    if rank == 0:
        # Print the results if master rank
        if solution is None:
            raise ValueError(f'Solution to problem {problem} not found!')
        
        solution.verbose = args.verbose
        print(solution)
        
        # Save solution
        if isinstance(args.output, str):
            output_path = args.output
        else:
            output_path = args.problem_instance + '.output'
        
        file_writer = MaxCutToJson()
        file_writer.write(solution,
                          output_path,
                          overwrite=args.overwrite,
                          with_metadata=True)
