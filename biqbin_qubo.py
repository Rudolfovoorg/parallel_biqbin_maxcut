from biqbin_base import QUBOSolver, QuboFromJson, QuboFromQPLIB, ParserQubo, QuboToJson

"""
    Default Qubo solver using Biqbin MaxCut wrapper
"""

if __name__ == '__main__':
    parser = ParserQubo()
    args = parser.parse_args()
    
    # Select the reader class based on the file format
    if args.qplib:
        file_reader = QuboFromQPLIB()
    else:
        file_reader = QuboFromJson()
    
    # Read the file and get the problem
    problem = file_reader.read(args.problem_instance, optimize_input=args.optimize)
    
    # Initialize QUBOSolver class which takes a path to parameters file and time limit
    solver = QUBOSolver(params=args.params, time_limit=args.time)
    
    # Run biqbin solver to solve the qubo, passing in the problem
    solution = solver.compute(problem=problem)

    # Get the MPI rank and if master rank print the solution and save it as json
    rank = solver.get_rank()
    if rank == 0:
        # Master rank prints the results
        if solution is None:
            raise ValueError(f'Could not compute solution for {problem}')
        print(solution.meta_data)
        
        # Save output path
        file_writer = QuboToJson()
        if isinstance(args.output, str):
            output_path = args.output
            print(output_path)
        else:
            output_path = args.problem_instance + '.output'
        file_writer.write(solution,
                          output_path,
                          overwrite=args.overwrite,
                          with_metadata=True,
                          with_maxcut_solution=True)
