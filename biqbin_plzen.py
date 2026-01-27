from biqbin.biqbin_base import QUBOSolver, init_mpi
from biqbin.data_parsers import QuboFromJson, QuboToJson
from biqbin.argparsers import ArgParserQubo
import numpy as np
import json
"""
    Default Qubo solver using Biqbin MaxCut wrapper
"""

if __name__ == '__main__':
    size, rank = init_mpi()
    
    parser = ArgParserQubo()
    args = parser.parse_args()

    file_reader = QuboFromJson(args.problem_instance, optimize_input=args.optimize)

    solution = None
    if rank == 0:
        with open(args.problem_instance[:-len('.json')] + '_solution.json') as f:
            solution = np.array(json.load(f)['x'])
            print(f'{solution = }')
    # Read the file and get the problem
    problem = file_reader.read()
    
    # Initialize QUBOSolver class which takes a path to parameters file and time limit
    solver = QUBOSolver(problem=problem, 
                        params=args.params,
                        time_limit=args.time,
                        initial_solution=solution)

    # Run biqbin solver to solve the qubo, passing in the problem
    solution = solver.compute()

    # Get the MPI rank and if master rank print the solution and save it as json
    if rank == 0:
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
