from biqbin.biqbin_base import QUBOSolver, get_rank
from biqbin.data_parsers import QuboFromJson, QuboToJson
from biqbin.argparsers import ArgParserQubo
import numpy as np
import json

"""
    Default Qubo solver using Biqbin MaxCut wrapper
"""

if __name__ == '__main__':
    parser = ArgParserQubo()
    args = parser.parse_args()

    file_reader = QuboFromJson(
        args.problem_instance, optimize_input=args.optimize)

    # Read the file and get the problem
    problem = file_reader.read()

    if get_rank() == 0 and args.solution:
        with open(args.solution, 'r') as f:
            initial_solution = np.array(json.load(f)['x'])
    else:
        initial_solution = None

    # Initialize QUBOSolver class which takes a path to parameters file and time limit
    solver = QUBOSolver(problem=problem,
                        params=args.params,
                        time_limit=args.time,
                        initial_solution=initial_solution,
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
        file_writer = QuboToJson(solution)

        if isinstance(args.output, str):
            output_path = args.output
        else:
            output_path = args.problem_instance + '.output'

        file_writer.write(output_path,
                          overwrite=args.overwrite,
                          with_metadata=True,
                          with_maxcut_solution=True
                          )
