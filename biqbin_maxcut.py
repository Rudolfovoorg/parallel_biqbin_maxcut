from biqbin import MaxCutSolver, get_rank, MaxCutSolutionToJson, init
from biqbin.argparsers import ArgParserMaxCut
import numpy as np
import json


if __name__ == '__main__':
    init()
    
    parser = ArgParserMaxCut()
    args = parser.parse_args()

    # Select the file reader based on the file format
    parser_class = args.format
    file_reader = parser_class(
        args.problem_instance,
        optimize_input=args.optimize)

    # Read the file
    problem = file_reader.read()

    if get_rank() == 0 and args.solution:
        with open(args.solution, 'r') as f:
            initial_estimate = np.array(json.load(f)['initial_estimate'])
    else:
        initial_estimate = None

    # Create an instance of the MaxCutSolver passing in path to params file and time limit
    solver = MaxCutSolver(
        problem=problem,
        params=args.params,
        time_limit=args.time,
        initial_estimate=initial_estimate,
        collect_heuristic_root_data=args.collect_root_data,
        collect_sdp_bound_root_data=args.collect_root_data)

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

        file_writer = MaxCutSolutionToJson(solution)
        file_writer.write(output_path,
                          overwrite=args.overwrite,
                          with_metadata=True)
