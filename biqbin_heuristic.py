import numpy as np
import sys
from neal import SimulatedAnnealingSampler
from biqbin_base import QUBOSolver, DataGetterJson, default_heuristic, ParserQubo, ParserDWaveHeuristic
import logging
from copy import deepcopy


logger = logging.getLogger(__name__)


class QuboDwaveSampler(QUBOSolver):
    def __init__(self, data_gettr, params: str, optimize_input:bool, time_limit: int, sampler, **sampler_kwargs):
        super().__init__(data_gettr, params, optimize_input, time_limit)
        self.sampler = sampler
        self.sampler_kwargs = sampler_kwargs

    def heuristic(self, L0: np.ndarray, L: np.ndarray, xfixed: np.array, sol_X: np.array, x: np.array):
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

        self.heuristic_counter += 1

        _x = np.array(
            list(self.sampler.sample_qubo(-L[:-1, :-1],
                 **self.sampler_kwargs).first.sample.values()),
            dtype=np.int32
        )

        _x = np.hstack([_x, [0]]) # simplification for above

        j = 0
        for i in range(len(x)):
            if xfixed[i] == 0:
                x[i] = _x[j]
                j += 1
            else:
                x[i] = sol_X[i]

        sol_value = self.evaluate_solution(L0, x)

        if logger.isEnabledFor(logging.DEBUG):
            her_value = default_heuristic(L0, L, xfixed, sol_X, deepcopy(x))
            logger.debug(
                f'Custom heuristic: {sol_value}, default heuristic: {her_value}')

        return sol_value

    def evaluate_solution(self, L0: np.ndarray, sol: np.ndarray) -> float:
        """Calculate the lowerbound value of heuristic solution

        Args:
            L0 (np.ndarray): main Problem *SP->L matrix
            sol (np.ndarray): current solution

        Returns:
            float: value of the solution
        """
        sol_val = 0
        for i in range(len(sol)):
            for j in range(len(sol)):
                sol_val += L0[i][j] * sol[i] * sol[j]
        return sol_val


if __name__ == '__main__':

    # https://stackoverflow.com/questions/7016056/python-logging-not-outputting-anything
    logging.basicConfig()

    parser = ParserDWaveHeuristic()
    argv = parser.parse_args()
    
    logging_level = logging.WARNING
    if argv.info:
        logging_level = logging.INFO
    if argv.debug:
        logging_level = logging.DEBUG
    logging.root.setLevel(logging_level)
    
    data_getter = DataGetterJson(argv.problem_instance)
    solver = QuboDwaveSampler(data_getter, 
                              params=argv.params, 
                              optimize_input=argv.optimize, 
                              time_limit=argv.time, 
                              sampler=SimulatedAnnealingSampler(), 
                              num_reads=10
                              )
    result = solver.run()

    rank = solver.get_rank()
    if logger.isEnabledFor(logging.INFO):
        print(f"{rank=} heuristics ran {solver.heuristic_counter} times")
        
    if rank == 0:
        # Master rank prints the results
        print(result)
        solver.save_result(result, argv.output, argv.overwrite)