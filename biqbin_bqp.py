"""Command-line entry point for equality-constrained binary quadratic problems."""

import json

import numpy as np

from biqbin import BQPSolver, BQPToJson, get_rank, init, logger
from biqbin.argparsers import ArgParserBQP
from biqbin.logging import configure_rank_logging

if __name__ == "__main__":
    parser = ArgParserBQP()
    args = parser.parse_args()

    init()  # initialize MPI
    configure_rank_logging(args.log_dir, get_rank())
    problem_reader_cls = args.format
    problem_reader = problem_reader_cls(args.problem_instance, optimize_input=args.optimize)
    problem = problem_reader.read()

    if get_rank() == 0 and args.solution:
        with open(args.solution, "r") as f:
            initial_estimate = np.array(json.load(f)["initial_estimate"])
        logger.info("Initial solution: %s", initial_estimate)
    else:
        initial_estimate = None

    solver = BQPSolver(
        problem,
        args.params,
        args.time,
        initial_estimate,
        collect_heuristic_root_data=args.collect_root_data,
        collect_sdp_bound_root_data=args.collect_root_data,
    )

    solution = solver.compute()  # run the solver

    if get_rank() == 0:
        if solution is None:
            raise ValueError(f"Solution to problem {problem} not found!")
        solution.print_computed_solution(args.verbose)
        solution_writer = BQPToJson(solution)
        if isinstance(args.output, str):
            output_path = args.output
        else:
            output_path = args.problem_instance + ".output"
        solution_writer.write(output_path, with_metadata=True, with_maxcut_solution=True)
