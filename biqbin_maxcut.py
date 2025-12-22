import argparse
from biqbin.biqbin_base import MaxCutSolver, ArgParserMaxCut, MaxCutToJson, get_rank
from biqbin.data_parsers import MaxCutFromEdgeWeights, MaxCutFromJson


if __name__ == '__main__':
    parser = ArgParserMaxCut()
    args = parser.parse_args()

    # Select the file reader based on the file format
    if args.edge_weight:
        file_reader = MaxCutFromEdgeWeights(
            args.problem_instance, optimize_input=args.optimize)
    else:
        file_reader = MaxCutFromJson(
            args.problem_instance, optimize_input=args.optimize)

    # Read the file
    problem = file_reader.read()

    # Create an instance of the MaxCutSolver passing in path to params file and time limit
    solver = MaxCutSolver(
        problem=problem,
        params=args.params,
        time_limit=args.time)
    # Compute the solution for the given problem
    solution = solver.compute()
    # Get rank to only save the results on the master rank
    if get_rank() == 0:
        # Print the results if master rank
        if solution is None:
            raise ValueError(f'Solution to problem {problem} not found!')

        solution.print_computed_solution(args.verbose)
        # Save solution
        if args.output:
            output_path = args.output
        else:
            output_path = args.problem_instance + '.output'

        file_writer = MaxCutToJson(solution)
        file_writer.write(output_path,
                          overwrite=args.overwrite,
                          with_metadata=True)
