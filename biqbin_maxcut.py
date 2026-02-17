import logging
from biqbin.biqbin_base import MaxCutSolver, get_rank, logger
from biqbin.data_parsers import MaxCutFromEdgeWeights, MaxCutFromJson, MaxCutToJson
from biqbin.argparsers import ArgParserMaxCut
import numpy as np
import json

if __name__ == '__main__':
    parser = ArgParserMaxCut()
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Select the file reader based on the file format
    if args.edge_weight:
        file_reader = MaxCutFromEdgeWeights(
            args.problem_instance, optimize_input=args.optimize)
    else:
        file_reader = MaxCutFromJson(
            args.problem_instance, optimize_input=args.optimize)

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
        collect_heuristic_data=args.collect_heur_data)

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
