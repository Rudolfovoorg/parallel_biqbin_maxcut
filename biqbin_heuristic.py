import numpy as np
from numpy import typing as npt
from dwave.samplers import SimulatedAnnealingSampler
from biqbin.biqbin_base import QUBOSolver, get_rank
from biqbin.data_parsers import QuboFromJson, QuboToJson
from biqbin.argparsers import ArgParserDWaveHeuristic
import logging
import json


class QuboDwaveSampler(QUBOSolver):
    def __init__(self, problem,
                 params: str,
                 time_limit: int,
                 initial_solution: np.ndarray | None,
                 collect_heuristic_data: bool,
                 sampler, **sampler_kwargs):
        super().__init__(problem, params, time_limit,
                         initial_solution, collect_heuristic_data)
        self.sampler = sampler
        self.sampler_kwargs = sampler_kwargs
        self.heuristic_counter = 0

    def heuristic(self, L: np.ndarray, **kwargs) -> npt.ArrayLike:
        """Heuristc with D-Waves simulated annealing sampler

        Args:
            L (np.ndarray): Subproblem Laplacean matrix

        Returns:
            np.ndarray: Solution [0, 1] binary vector of size L.shape[0] - 1
        """
        x = np.array(
            list(self.sampler.sample_qubo(-L[:-1, :-1],
                 **self.sampler_kwargs).first.sample.values()),
            dtype=np.int32
        )
        return x


if __name__ == '__main__':
    parser = ArgParserDWaveHeuristic()
    args = parser.parse_args()

    reader = args.format(args.problem_instance, optimize_input=args.optimize)
    problem = reader.read()
    if get_rank() == 0 and args.solution:
        with open(args.solution, 'r') as f:
            initial_solution = np.array(json.load(f)['initial_estimate'])
    else:
        initial_solution = None

    solver = QuboDwaveSampler(problem=problem,
                              params=args.params,
                              time_limit=args.time,
                              initial_solution=initial_solution,
                              collect_heuristic_data=args.collect_heur_data,
                              sampler=SimulatedAnnealingSampler(),
                              num_reads=10)

    solution = solver.compute()
    rank = get_rank()

    if rank == 0:
        # Master rank prints the results
        if solution is None:
            raise ValueError(f'Solution to problem {problem} not found!')

        solution.print_computed_solution(args.verbose)
        solution_writer = QuboToJson(solution)
        if isinstance(args.output, str):
            output_path = args.output
        else:
            output_path = args.problem_instance + '.output'

        solution_writer.write(output_path,
                              overwrite=args.overwrite,
                              with_maxcut_solution=True)
