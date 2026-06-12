import json
import numpy.typing as npt
import numpy as np

from biqbin import QUBOSolver, ProblemQubo, QuboSolutionToJson, init, get_rank, logger
from biqbin.argparsers import ArgParserQubo
from biqbin.biqbin_module import BabNode, Problem, reduce_sum_mpi

"""
Bellow are mix-and-match classes built for testing purposes.

- Custom root_sdp_bound returns 1000.
- Custom sdp_bound returns 1e15 and will explore the entire B&B tree.
- Custom root_heuristic returns an all ones vector.
- Custom heuristic returns an all zero vector.
- Custom SDP primal solution matrix is set to identity matrix.
"""


class TestTrackerSolver(QUBOSolver):
    def __init__(self, problem: ProblemQubo, params: str = 'params',
                 time_limit: int = 0,
                 initial_estimate: np.ndarray | None = None,
                 collect_heur_root_data: bool = False,
                 collect_sdp_root_data: bool = False):
        super().__init__(problem, params, time_limit, initial_estimate,
                         collect_heur_root_data, collect_sdp_root_data)

        # Call counts of custom methods
        self.custom_root_sdp_call_count = 0
        self.custom_sdp_call_count = 0
        self.custom_root_heuristic_call_count = 0
        self.custom_heuristic_call_count = 0

    def _update_result_dict(self, raw_result: dict) -> dict | None:
        updated_results = super()._update_result_dict(raw_result)
        self.custom_sdp_call_count = reduce_sum_mpi(self.custom_sdp_call_count)
        self.custom_heuristic_call_count = reduce_sum_mpi(
            self.custom_heuristic_call_count)
        if updated_results:
            updated_results['meta_data']['custom_solver_tests'] = {
                'root_sdp_calls': self.custom_root_sdp_call_count,
                'sdp_calls': self.custom_sdp_call_count,
                'root_heuristic_calls': self.custom_root_heuristic_call_count,
                'heuristic_calls': self.custom_heuristic_call_count,
            }
        return updated_results


def custom_root_sdp_bound(self, node: BabNode, P0: Problem, P: Problem, *args, **kwargs) -> float:
    """Custom SDP bound routine on root B&B node.

    Before returning we call ``set_sdp_primal_solution`` passing in a
    PSD matrix of the same size as subproblem Laplacean (P.L).
    """
    self.custom_root_sdp_call_count += 1

    # P0.L and P.L are of the same shape on the root node
    X = np.identity(P0.n)
    self.set_sdp_primal_solution(X)

    sdp_value: float = 1000
    return sdp_value


def custom_sdp_bound(self, node: BabNode, P0: Problem, P: Problem, *args, **kwargs) -> float:
    """Custom SDP bound routine.

    Before returning we call ``set_sdp_primal_solution`` passing in a
    PSD matrix of the same size as subproblem Laplacean (P.L).
    """
    self.custom_sdp_call_count += 1

    # P0.L and P.L are of the same shape on the root node
    X = np.identity(P.n)
    self.set_sdp_primal_solution(X)

    sdp_value: float = 1e15
    return sdp_value


def custom_root_heuristic(self, L: np.ndarray, *args, **kwargs) -> npt.NDArray[np.int32]:
    """Custom heuristic on root B&B nodes. 

    If ``root_heuristic`` is not overwritten, this function will also be used on the root node.
    """
    self.custom_root_heuristic_call_count += 1
    x: npt.ArrayLike = np.ones(kwargs['P'].n - 1, dtype=np.int32)
    return x


def custom_heuristic(self, L: np.ndarray, *args, **kwargs) -> npt.NDArray[np.int32]:
    """Custom heuristic on non-root B&B nodes. 

    If ``root_heuristic`` is not overwritten, this function will also be used on the root node.
    """
    self.custom_heuristic_call_count += 1
    x: npt.ArrayLike = np.zeros(kwargs['P'].n - 1, dtype=np.int32)
    return x


CUSTOM_CONFIGS = [
    (None, None, None, custom_root_sdp_bound),
    (None, None, custom_sdp_bound, None),
    (None, None, custom_sdp_bound, custom_root_sdp_bound),
    (None, custom_root_heuristic, None, None),
    (None, custom_root_heuristic, None, custom_root_sdp_bound),
    (None, custom_root_heuristic, custom_sdp_bound, None),
    (None, custom_root_heuristic, custom_sdp_bound, custom_root_sdp_bound),
    (custom_heuristic, None, None, None),
    (custom_heuristic, None, None, custom_root_sdp_bound),
    (custom_heuristic, None, custom_sdp_bound, None),
    (custom_heuristic, None, custom_sdp_bound, custom_root_sdp_bound),
    (custom_heuristic, custom_root_heuristic, None, None),
    (custom_heuristic, custom_root_heuristic, None, custom_root_sdp_bound),
    (custom_heuristic, custom_root_heuristic, custom_sdp_bound, None),
    (custom_heuristic, custom_root_heuristic,
     custom_sdp_bound, custom_root_sdp_bound),
]


def configure_solver(index: int):
    (heur, root_heur, sdp, root_sdp) = CUSTOM_CONFIGS[index]
    if heur:
        TestTrackerSolver.heuristic = heur
    if root_heur:
        TestTrackerSolver.root_heuristic = root_heur
    if sdp:
        TestTrackerSolver.sdp_bound = sdp
    if root_sdp:
        TestTrackerSolver.root_sdp_bound = root_sdp


class ArgParserCustom(ArgParserQubo):
    def __init__(self, prog='biqbin_custom_example.py', description='Biqbin Custom solver'):
        super().__init__(prog, description)
        self.add_argument('--test-case',
                          type=int,
                          choices=range(len(CUSTOM_CONFIGS)),
                          help=f'Custom solvers')


if __name__ == '__main__':
    init()
    parser = ArgParserCustom()
    args = parser.parse_args()

    parser_class = args.format
    file_reader = parser_class(
        args.problem_instance, optimize_input=args.optimize)

    # Read the file and get the problem
    problem = file_reader.read()

    if get_rank() == 0 and args.solution:
        with open(args.solution, 'r') as f:
            initial_estimate = np.array(json.load(f)['initial_estimate'])
    else:
        initial_estimate = None

    configure_solver(args.test_case)

    # Initialize QUBOSolver class which takes a path to parameters file and time limit
    solver = TestTrackerSolver(problem=problem,
                               params=args.params,
                               time_limit=args.time,
                               initial_estimate=initial_estimate,
                               collect_heur_root_data=args.collect_root_data,
                               collect_sdp_root_data=args.collect_root_data
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
        file_writer = QuboSolutionToJson(solution)

        if isinstance(args.output, str):
            output_path = args.output
        else:
            output_path = args.problem_instance + f'.output.json'

        file_writer.write(output_path,
                          overwrite=args.overwrite,
                          with_metadata=True,
                          with_maxcut_solution=True
                          )
