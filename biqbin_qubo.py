from biqbin import QUBOSolver, get_rank, QuboSolutionToJson
from biqbin.argparsers import ArgParserQubo
import numpy as np
import json

from biqbin.parameters import BiqbinParameters

"""
    Default Qubo solver using Biqbin MaxCut wrapper
"""

if __name__ == '__main__':
    parser = ArgParserQubo()
    args = parser.parse_args()

    parser_class = args.format
    file_reader = parser_class(
        args.problem_instance, optimize_input=args.optimize)

    # Read the file and get the problem
    problem = file_reader.read()

    if get_rank() == 0 and args.solution:
        with open(args.solution, 'r') as f:
            initial_estimate = np.array(json.load(f)['initial_estimate'])
    else:
        initial_estimate = None

    params = BiqbinParameters.from_toml(args.params)
    params.root = args.root
    params.time_limit = args.time

    # Initialize QUBOSolver class which takes a path to parameters file and time limit
    solver = QUBOSolver(problem=problem,
                        params=params,
                        initial_estimate=initial_estimate,
                        collect_heur_data=args.collect_heur_data
                        )

    # Run biqbin solver to solve the qubo, passing in the problem
    solution = solver.compute()

    # Get the MPI rank and if master rank print the solution and save it as json
    if get_rank() == 0:
        # Master rank prints the results
        if solution is None:
            raise ValueError(f'Could not compute solution for {problem}')

        solution.print_computed_solution(args.verbose)
        # Save output path
        file_writer = QuboSolutionToJson(solution)

        if isinstance(args.output, str):
            output_path = args.output
        else:
            output_path = args.problem_instance + '.output'

        file_writer.write(output_path,
                          overwrite=args.overwrite,
                          with_metadata=True,
                          with_maxcut_solution=True
                          )
