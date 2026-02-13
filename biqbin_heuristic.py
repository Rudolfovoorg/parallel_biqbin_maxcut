import numpy as np
from dwave.samplers import SimulatedAnnealingSampler
from biqbin.biqbin_base import QUBOSolver, goemans_williamson_heuristic, get_rank
from biqbin.data_parsers import QuboFromJson, QuboToJson
from biqbin.argparsers import ArgParserDWaveHeuristic
import logging
import json
from copy import deepcopy


logger = logging.getLogger(__name__)


class QuboDwaveSampler(QUBOSolver):
    def __init__(self, problem, 
                 params: str, 
                 time_limit: int, 
                 initial_solution: np.ndarray | None, 
                 collect_heuristic_data: bool,
                 sampler, **sampler_kwargs):
        super().__init__(problem, params, time_limit, initial_solution, collect_heuristic_data)
        self.sampler = sampler
        self.sampler_kwargs = sampler_kwargs
        self.heuristic_counter = 0

    def heuristic(self, L0: np.ndarray, L: np.ndarray, xfixed: np.ndarray, sol_X: np.ndarray, x: np.ndarray):
        """Heuristc with D-Waves simulated annealing sampler

        Args:
            L0 (np.ndarray): main Problem *SP->L matrix
            L (np.ndarray): subproblem *PP->L matrix
            xfixed (np.array): BabNode xfixed variables array
            sol_X (np.array): Solution.X array in BabNode
            x (np.array): stores current best solution

        Returns:
            np.ndarray: solution nodes provided by the heuristc, should be in 0, 1 form (1 node is chosen, 0 it is not chosen)
        """

        _x = np.array(
            list(self.sampler.sample_qubo(-L[:-1, :-1],
                 **self.sampler_kwargs).first.sample.values()),
            dtype=np.int32
        )
        
        free_mask = (xfixed == 0)
        np.copyto(x, sol_X)
        x[free_mask] = _x

        sol_value = self._evaluate_solution(L0, x)

        if logger.isEnabledFor(logging.DEBUG):
            j = 0
            x_copy = np.zeros(x.size)
            for i in range(len(x)):
                if xfixed[i] == 0:
                    x_copy[i] = _x[j]
                    j += 1
                else:
                    x_copy[i] = sol_X[i]
                    
            vectorized_equal_to_org = np.array_equal(x, x_copy)
            if not vectorized_equal_to_org:
                raise ValueError(f'x mismatch!:\n{x =}\n{x_copy = }')
            
            her_value = goemans_williamson_heuristic(
                L0, L, xfixed, sol_X, deepcopy(x))
            logger.debug(
                f'Custom heuristic: {sol_value}, default heuristic: {her_value}, {vectorized_equal_to_org}')

        return sol_value

if __name__ == '__main__':

    # https://stackoverflow.com/questions/7016056/python-logging-not-outputting-anything
    logging.basicConfig()

    parser = ArgParserDWaveHeuristic()
    args = parser.parse_args()

    logging_level = logging.WARNING
    if args.info:
        logging_level = logging.INFO
    if args.debug:
        logging_level = logging.DEBUG
    logging.root.setLevel(logging_level)

    reader = QuboFromJson(args.problem_instance, optimize_input=args.optimize)
    problem = reader.read()
    if get_rank() == 0 and args.solution:
        with open(args.solution, 'r') as f:
            initial_solution = np.array(json.load(f)['x'])
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
    if logger.isEnabledFor(logging.INFO):
        print(f"{rank=} heuristics ran {solver.heuristic_counter} times")

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
