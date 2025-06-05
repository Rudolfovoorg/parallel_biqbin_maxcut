import numpy as np
import sys
from neal import SimulatedAnnealingSampler
from biqbin_base import QUBOSolver, DataGetterJson, default_heuristic
import logging
from copy import deepcopy


logger = logging.getLogger(__name__)


class QuboSimulatedAnnealing(QUBOSolver):
    def __init__(self, data_gettr, params: str, num_reads: int = 10):
        super().__init__(data_gettr, params)
        self.sampler = SimulatedAnnealingSampler()
        self.num_reads = num_reads

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
                 num_reads=self.num_reads).first.sample.values()),
            dtype=np.int32
        )

        #  This can be likely simplified.
        # _x = 2*_x-1                # normalize between -1, 1
        # _x = np.hstack([_x, [-1]]) # append -1
        # _x = 1/2*(_x+1)            # normalize back to 0, 1

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
    logging.root.setLevel(logging.DEBUG)

    _, problem_instance_file_name, params = sys.argv

    data_getter = DataGetterJson(problem_instance_file_name)

    solver = QuboSimulatedAnnealing(data_getter, params=params, num_reads=5)
    result = solver.run()

    rank = solver.get_rank()
    print(f"{rank=} heuristics ran {solver.heuristic_counter} times")
    if rank == 0:
        # Master rank prints the results
        print(result)
        solver.save_result(result)