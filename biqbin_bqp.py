import json
import scipy as sp
import numpy as np
import warnings

from biqbin.utils import convert_numpy_to_json_serializable
from biqbin.biqbin_base import MaxCutSolver, SolutionMaxCut, ProblemMaxCut, get_rank
from biqbin.argparsers import ArgParserBase
from biqbin.data_parsers import FromFile, ToFile

# these functions are placeholder implementations!
from biqbin.bqp_data_processing_PLACEHOLDER import read_data_bqp, read_data_bqp_json, read_solution_bqp


class ParserBQP(ArgParserBase):
    def __init__(self):
        super().__init__(prog=f'biqbin_bqp.py', description='Biqbin BQP solver')
        self.add_argument('-j', '--json', action='store_true',
                          help='use json input file')


class ProblemBQP(ProblemMaxCut):
    def __init__(self, maxcut_adjacency_matrix: np.ndarray, problem_name: str, optimize_input: bool) -> None:
        super().__init__(maxcut_adjacency_matrix, problem_name, optimize_input)


class SolutionBQP(SolutionMaxCut):
    def __init__(self, biqbin_result: dict, problem: ProblemBQP) -> None:
        super().__init__(biqbin_result, problem)
        self.maxcut_solution = super().solution
        self.__solution = read_solution_bqp(
            biqbin_result, len(super().solution['x']) - 1)

    @property
    def solution(self):
        return self.__solution

    def __str__(self) -> str:
        return (f'{super().__str__()}\n'
                f'--- BQP ---\n'
                f' Computed value = {self.__solution['computed_val']}\n'
                f'       Feasible = {self.__solution['feasible_solution']}\n'
                f'              x = {self.__solution['x']}\n'
                f'            Rho = {self.__solution['rho']}\n'
                f'    Const value = {self.__solution['const_value']}\n')


class BQPSolver(MaxCutSolver):
    solver_name = f'PyBiqBin-BQP-PLACEHOLDER'

    def __init__(self, problem: ProblemBQP, params: str, time_limit: int = 0, initial_solution=None, collect_heuristic_data=False):
        super().__init__(problem=problem, params=params, time_limit=time_limit,
                         initial_solution=initial_solution, collect_heuristic_data=collect_heuristic_data)
        self.__problem: ProblemBQP = problem

    @property
    def problem(self) -> ProblemBQP:
        return self.__problem

    def compute(self) -> SolutionBQP | None:
        result = self._run_solver()
        if result is not None:
            print(result)
            return SolutionBQP(result, self.problem)


class BQPFromFile(FromFile):
    """
    Uses the C implementation of biqbin_general_bqp, reads and parses bqp instance file and parses
    into the adjacency matrix.
    """

    def read(self) -> ProblemBQP:
        warnings.warn("PLACEHOLDER FUNCTION")
        # This needs to be done in Python properly
        adj_matrix = read_data_bqp(self.filename)

        return ProblemBQP(adj_matrix, self.problem_name, self.optimize_input)


class BQPFromJson(FromFile):
    """
    Uses the default json implementation of biqbin_general_bqp, reads and parses bqp instance file and parses
    into the adjacency matrix.
    """

    def read(self) -> ProblemBQP:
        warnings.warn("PLACEHOLDER FUNCTION")
        instance = self.read_bqp_json(self.filename)
        adj_matrix = read_data_bqp_json(instance)

        return ProblemBQP(adj_matrix, self.problem_name, self.optimize_input)

    def read_bqp_json(self, filename):
        with open(filename, 'r') as file:
            instance = json.load(file)
        # zes it is realy like this in biqbin :(

        def f(F):
            for i, j, v in F:
                if i == j:
                    yield (i, j), v
                else:
                    yield (i, j), v
                    yield (j, i), v

        Fdict = dict(f(instance["F"]))
        Anp = np.array(instance["A"])
        cnp = np.array(instance["c"])
        bnp = np.array(instance["b"])

        Find, Fv = (list(Fdict.keys()), list(Fdict.values()))
        Find = np.asarray(Find)

        Fm = sp.sparse.coo_matrix((Fv, (Find[:, 0], Find[:, 1])), shape=(
            instance["number_of_variables"], instance["number_of_variables"])).todense()
        Am = sp.sparse.coo_matrix((Anp[:, 2], (Anp[:, 0], Anp[:, 1])), shape=(
            instance["number_of_constraints"], instance["number_of_variables"])).todense()
        cm = sp.sparse.coo_matrix((cnp[:, 1], ([0]*len(instance["c"]), cnp[:, 0])),
                                  shape=(1, instance["number_of_variables"])).todense()
        bm = sp.sparse.coo_matrix((bnp[:, 1], (bnp[:, 0], [
                                  0]*len(instance["b"]))), shape=(instance["number_of_constraints"], 1)).todense()

        instance["Fm"] = Fm
        instance["Am"] = Am
        instance["cm"] = cm
        instance["bm"] = bm

        return instance


class BQPToJson(ToFile):
    def __init__(self, solution: SolutionBQP) -> None:
        self.solution: SolutionBQP = solution

    def write(self, filename: str, overwrite: bool = False, with_metadata: bool = True, with_maxcut_solution: bool = False) -> None:
        """Save the bqp solution as JSON file.

        Args:
            solution (SolutionBQP): Solution class returned by Biqbin after solving the problem
            with_metadata (bool, optional): Add meta_data to output file. Defaults to True.
        """

        # Check if output filename exists if we are not overriding and replace with filename_N.json
        output_path = self.get_output_path(filename, overwrite)

        save_output = {
            'bqp': self.solution.solution,
        }
        if with_maxcut_solution:
            save_output['maxcut'] = self.solution.maxcut_solution
        if with_metadata:
            save_output['meta_data'] = self.solution.meta_data

        with open(output_path, 'w') as f:
            json.dump(save_output, f,
                      default=convert_numpy_to_json_serializable)


if __name__ == '__main__':
    parser = ParserBQP()
    args = parser.parse_args()

    # Get the file reader for the BQP instance
    if args.json:
        problem_reader = BQPFromJson(
            args.problem_instance, optimize_input=args.optimize)
    else:
        problem_reader = BQPFromFile(
            args.problem_instance, optimize_input=args.optimize)

    problem = problem_reader.read()
    
    if args.solution:
        with open(args.solution, 'r') as f:
            initial_solution = np.array(json.load(f)['x'])
    else:
        initial_solution = None
    
    solver = BQPSolver(problem, args.params, args.time, initial_solution)

    solution = solver.compute()  # run the solver

    if get_rank() == 0:
        # Convert the Max-Cut solution back to BQP !! PLACEHOLDER FUNCTION !!
        if solution is None:
            raise ValueError(f'Solution to problem {problem} not found!')
        print(solution)
        solution_writer = BQPToJson(solution)
        if isinstance(args.output, str):
            output_path = args.output
        else:
            output_path = args.problem_instance + '.output'
        solution_writer.write(output_path,
                              with_metadata=True,
                              with_maxcut_solution=True)
