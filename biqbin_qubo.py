from biqbin_base import QUBOSolver, QuboFromJson, QuboFromQPLIB, ParserQubo, QuboToJson

"""
    Default Qubo solver using Biqbin MaxCut wrapper
"""

if __name__ == '__main__':
    # Path to qubo json file file and path to parameters file
    parser = ParserQubo()
    args = parser.parse_args()
    
    if args.qplib:
        file_loader = QuboFromQPLIB(args.problem_instance)
    else:
        file_loader = QuboFromJson(args.problem_instance)
    
    problem = file_loader.read()
    # Initialize QUBOSolver class which takes a DataGetter class instance, path to parameters file and bool if optimizing
    solver = QUBOSolver(problem=problem, params=args.params, time_limit=args.time)
    # Run biqbin solver
    solution = solver.run()

    rank = solver.get_rank()
    if rank == 0:
        # Master rank prints the results
        if solution is None:
            raise ValueError(f'Could not compute solution for {problem}')
        print(solution.meta_data)
        QuboToJson(args.problem_instance + '.output', args.overwrite).write(solution)
