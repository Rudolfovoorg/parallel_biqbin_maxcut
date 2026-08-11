import json
from typing import Any
import warnings
import numpy as np
from typing import Literal
from argparse import ArgumentTypeError, SUPPRESS

from biqbin.utils import convert_numpy_to_json_serializable, check_matrix_validity
from biqbin import MaxCutSolver, SolutionMaxCut, ProblemMaxCut, get_rank, init, logger
from biqbin.argparsers import ArgParserBase
from biqbin.data_parsers import FromFile, ToFile
from biqbin.biqbin_module import interior_point_method_maxcut, abort_mpi

FileDataSection = Literal['F', 'c', 'A', 'b']


class ProblemBQP(ProblemMaxCut):
    """
    Linearly constrained binary quadratic optimization problem.

    ```
    min         x.T @ F @ x + c.T @ x
    subject to  A @ x = b
                x_i in {0, 1}
    ```
    """

    def __init__(self, F: np.ndarray, c: np.ndarray, A: np.ndarray, b: np.ndarray, problem_name: str, optimize_input: bool):
        """
        All values in the inputs `F`, `c`, `A` and `b` need to have integer values.

        ```
        min         x.T @ F @ x + c.T @ x
        subject to  A @ x = b
                    x_i in {0, 1}
        ```

        Args:
            F (np.ndarray): Quadratic objective matrix. It is expected to be a symmetric matrix.
            c (np.ndarray): Linear objective vector.
            A (np.ndarray): Equality-constraint matrix.
            b (np.ndarray): Equality-constraint vector.
            problem_name (str): Human-readable name of the problem.
            optimize_input (bool): Optimizes the input for the solver.
        """
        self.F: np.ndarray = check_matrix_validity(F)
        self.c: np.ndarray = check_matrix_validity(c, False)
        self.A: np.ndarray = check_matrix_validity(A, False)
        self.b: np.ndarray = check_matrix_validity(b, False)

        maxcut_adj, const_val, penalty, rho = self.process_bqp_input()

        self.penalty: float = penalty
        self.rho: float = rho
        self.const_value: float = const_val
        logger.info(f'Penalty        = {penalty}')
        logger.info(f'Rho            = {rho}')
        logger.info(f'Constant value = {const_val}')

        super().__init__(maxcut_adj, problem_name, optimize_input)

    def process_bqp_input(self) -> tuple[np.ndarray, int, int, float]:
        """
        Transform a linearly constrained BQP into a Max-Cut adjacency matrix.

        Assumes:
            ipm_mc_pk(C) -> maximum value of <C, X>
            subject to diag(X) = 1, X >= 0

        Returns:
            adj (np.ndarray): Max-Cut adjacency matrix
            const_val (int):  Recover original value with: original_value = const_val - max_cut_value
            penalty (int):    Exact penalty parameter.
            rho (float):      Max absolute SDP solution value
        """
        n = self.F.shape[0]
        e = np.ones(n)

        # Change problem variables from {0, 1} to {-1, 1}
        sdp_constant = 0.25 * e @ self.F @ e + 0.5 * self.c @ e

        # Computing the penalty parameter
        A_scaled = 0.5 * self.A
        b_shifted = self.b - A_scaled @ e
        c_shifted = 0.5 * (self.F @ e + self.c)
        F_scaled = 0.25 * self.F

        # SDP matrix
        C = np.block([
            [F_scaled,                 0.5 * c_shifted[:, None]],
            [0.5 * c_shifted[None, :], np.array([[sdp_constant]])],
        ])

        # SDP maximum and minimum.
        r_max, _ = interior_point_method_maxcut(C)
        r_min, _ = interior_point_method_maxcut(-C)

        rho = max(abs(r_min), abs(r_max))
        penalty = int(np.ceil(2 * rho + 1))

        # Penalized quadratic matrix.
        top_left = F_scaled + penalty * A_scaled.T @ A_scaled
        top_right = 0.5 * c_shifted - penalty * A_scaled.T @ b_shifted
        bottom_right = sdp_constant + penalty * (b_shifted @ b_shifted)

        B = np.block([
            [top_left,                 top_right[:, None]],
            [top_right[None, :],       np.array([[bottom_right]])],
        ])

        # original_value = const_val - max_cut_value
        #
        # Derivation: maxcut_adjacency = 4*B with zero diagonal, and for such a
        # zero-diagonal weight matrix M, cut_value = 1/4 * (M.sum() - y.T @ M @ y)
        # where y = (x, 1).
        #
        # Substituting M = 4*B (diagonal cancels since y_i**2 == 1) gives
        # y.T @ B @ y == B.sum() - cut_value, i.e. B.sum() is exactly the
        # constant to recover the original objective.

        const_val = B.sum()
        if not np.isclose(const_val, round(const_val)):
            raise ValueError(f'Constant must be an integer, got {const_val}')
        const_val = round(const_val)

        # Standard weighted Max-Cut adjacency matrix.
        maxcut_adjacency = 4 * B
        np.fill_diagonal(maxcut_adjacency, 0)

        return maxcut_adjacency, const_val, penalty, rho


class SolutionBQP(SolutionMaxCut):
    def __init__(self, biqbin_result: dict, problem: ProblemBQP) -> None:
        super().__init__(biqbin_result, problem)
        self.maxcut_solution = super().solution
        self.__solution = self.get_bqp_solution(biqbin_result, problem)

    @property
    def solution(self):
        return self.__solution

    def get_bqp_solution(self, biqbin_result: dict, problem: ProblemBQP) -> dict[str, Any]:
        bqp_result: dict[str, Any] = {'rho': problem.rho,
                                      'const_value': problem.const_value,
                                      # TODO: add penalty when expanding tests!!! 'penalty': problem.penalty
                                      }
        biqbin_result['bqp'] = bqp_result
        mc_obj_value = biqbin_result['maxcut']['computed_val']
        # Max-Cut solution vector is larger by 1 than BQP, this last MC elemnt is always 0
        mc_x = np.array(biqbin_result['maxcut']['x'])[:-1]

        bqp_obj_value = problem.const_value - mc_obj_value
        if bqp_obj_value > problem.rho:
            bqp_result['feasible_solution'] = False
            return bqp_result

        bqp_result['feasible_solution'] = True
        bqp_result['computed_val'] = bqp_obj_value

        opt_value = mc_x @ problem.F @ mc_x + mc_x @ problem.c
        if np.isclose(opt_value, bqp_result['computed_val']):
            bqp_x = mc_x.tolist()
        else:
            bqp_x = (1 - mc_x).tolist()

        solution_check = bqp_x @ problem.F @ bqp_x + bqp_x @ problem.c
        if not np.isclose(solution_check, bqp_obj_value):
            raise ValueError(
                'Solution vector does not match the computed solution')

        bqp_result['x'] = bqp_x
        return bqp_result

    def __str__(self) -> str:
        return (f'{super().__str__()}\n'
                f'--- BQP ---\n'
                f' Computed value = {self.__solution["computed_val"]}\n'
                f'       Feasible = {self.__solution["feasible_solution"]}\n'
                f'              x = {self.__solution["x"]}\n'
                f'            Rho = {self.__solution["rho"]}\n'
                f'    Const value = {self.__solution["const_value"]}\n')


class BQPSolver(MaxCutSolver):
    solver_name = 'PyBiqBin-BQP-PLACEHOLDER'

    def __init__(self, problem: ProblemBQP, params: str, time_limit: int = 0, initial_estimate=None, collect_heuristic_root_data=False, collect_sdp_bound_root_data=False):
        self.__problem: ProblemBQP = problem

        if initial_estimate is not None:
            # Transform BQP solution to Biqbin Max-Cut solution vector, throwing error if not feasible
            initial_estimate = self._initial_estimate_bqp_to_maxcut(
                initial_estimate)

        super().__init__(problem=problem,
                         params=params,
                         time_limit=time_limit,
                         initial_estimate=initial_estimate,
                         collect_heuristic_root_data=collect_heuristic_root_data,
                         collect_sdp_bound_root_data=collect_sdp_bound_root_data)

    @property
    def problem(self) -> ProblemBQP:
        return self.__problem

    def compute(self) -> SolutionBQP | None:
        result = self._run_solver()
        if result is not None:
            return SolutionBQP(result, self.problem)

    def _initial_estimate_bqp_to_maxcut(self, bqp_x):
        """Transforms the BQP solution vector to a Max-Cut vector.

        Args:
            bqp_x (np.ndarray): BQP binary vector solution

        Raises:
            ValueError: bqp_x must be a feasable solution to the problem

        Returns:
            np.ndarray: Max-Cut solution vector
        """
        bqp_x = np.asarray(bqp_x)
        bqp_obj = bqp_x @ self.__problem.F @ bqp_x + self.__problem.c @ bqp_x

        for candidate in (bqp_x, 1 - bqp_x):
            # append the fixed auxiliary coordinate
            mc_candidate = np.append(candidate, 0)
            y = 1 - 2 * mc_candidate  # 0/1 partition -> {-1, 1}
            cut_value = 0.25 * (self.__problem.maxcut_adjacency_matrix.sum() -
                                y @ self.__problem.maxcut_adjacency_matrix @ y)

            if np.isclose(problem.const_value - cut_value, bqp_obj):
                return mc_candidate

        logger.fatal(
            'initial estimate solution is not a feasible solution to the problem.\n'
            'Please input a feasible initial estimate or run without passing one in.', stack_info=True)
        abort_mpi(10)


class BQPFromBQPFile(FromFile):
    """
    Read the custom BQP file format, defined in https://github.com/Rudolfovoorg/parallel_biqbin_maxcut/blob/main/doc/BQP_INPUT_EXAMPLE.md
    """

    def read(self) -> ProblemBQP:
        with open(self.filename, 'r') as f:
            lines = f.readlines()

        if not lines:
            raise ValueError(f'File {self.filename} is empty!')

        try:
            n, m = map(int, lines[0].split())

        except ValueError:
            raise ValueError(
                f'File {self.filename}: first line must be "n m", got: {lines[0]!r}'
            )

        F = np.zeros((n, n))
        c = np.zeros(n)
        A = np.zeros((m, n))
        b = np.zeros(m)

        current_section: FileDataSection | None = None

        def check_index(name: str, i: int, upper: int) -> None:
            if not (1 <= i <= upper):
                raise ValueError(
                    f'{name} index {i} out of range (must be 1..{upper})')

        for lineno, raw_line in enumerate(lines[1:], start=2):
            line = raw_line.strip()
            if not line:
                continue
            if line in {'F', 'c', 'A', 'b'}:
                current_section = line  # type: ignore
                continue
            if current_section is None:
                raise ValueError(
                    f'File {self.filename}, line {lineno}: data before any section header: {line!r}'
                )
            try:
                match current_section:
                    case 'F':
                        i, j, v = map(int, line.split())
                        check_index('F row', i, n)
                        check_index('F col', j, n)
                        i, j = i - 1, j - 1
                        F[i, j] = v
                        if i != j:
                            F[j, i] = v
                    case 'c':
                        i, v = map(int, line.split())
                        check_index('c', i, n)
                        c[i - 1] = v
                    case 'A':
                        i, j, v = map(int, line.split())
                        check_index('A row', i, m)
                        check_index('A col', j, n)
                        A[i - 1, j - 1] = v
                    case 'b':
                        i, v = map(int, line.split())
                        check_index('b', i, m)
                        b[i - 1] = v

            except ValueError as e:
                raise ValueError(
                    f'File {self.filename}, line {lineno} (section {current_section!r}): {e}\n  {line!r}'
                ) from e

        return ProblemBQP(F, c, A, b, self.problem_name, self.optimize_input)


class BQPFromJson(FromFile):
    """
    Reads and parses BQP JSON file.
    """

    def read(self) -> ProblemBQP:
        """Read the bqp json file and return a ProblemBQP class instance that SolverBQP can read.
        """
        with open(self.filename, 'r') as file:
            instance = json.load(file)

        n = instance["number_of_variables"]
        m = instance["number_of_constraints"]

        F = np.zeros((n, n))
        for i, j, v in instance["F"]:
            F[i, j] = v
            if i != j:
                F[j, i] = v

        A = np.zeros((m, n))
        for i, j, v in instance["A"]:
            A[i, j] = v

        c = np.zeros(n)
        for i, v in instance["c"]:
            c[i] = v

        b = np.zeros(m)
        for i, v in instance["b"]:
            b[i] = v

        return ProblemBQP(F, c, A, b, self.problem_name, self.optimize_input)


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


class ParserBQP(ArgParserBase):
    FORMAT_CHOICES = {
        'json': BQPFromJson,
        'bqp': BQPFromBQPFile
    }

    def __init__(self):
        super().__init__(prog='biqbin_bqp.py', description='Biqbin BQP solver')
        self.add_argument('-j', '--json', action='store_true',
                          help=SUPPRESS)
        self.add_argument(
            '--format',
            default=self.FORMAT_CHOICES["bqp"],
            type=self.parse_format,
            help=f'BQP problem instance file format. Valid formats are {tuple(self.FORMAT_CHOICES.keys())}; Defaults to \'bqp\'')

    def parse_args(self, *args, **kwargs):
        ns = super().parse_args(*args, **kwargs)
        if ns.json:
            warnings.warn(
                "[DEPRECATED] `-j`, `--json` is deprecated, please use --format json",
                UserWarning
            )
            ns.format = BQPFromJson
        return ns

    def parse_format(self, fmt: str) -> FromFile:
        fmt = fmt.lower()
        if fmt not in self.FORMAT_CHOICES:
            raise ArgumentTypeError(
                f'invalid format: {fmt}, choose from {tuple(i for i in self.FORMAT_CHOICES.keys())}')

        return self.FORMAT_CHOICES[fmt]


if __name__ == '__main__':
    init()
    parser = ParserBQP()
    args = parser.parse_args()

    problem_reader_cls = args.format
    problem_reader = problem_reader_cls(
        args.problem_instance, optimize_input=args.optimize)
    problem = problem_reader.read()

    if get_rank() == 0 and args.solution:
        with open(args.solution, 'r') as f:
            initial_estimate = np.array(json.load(f)['initial_estimate'])
        print(initial_estimate)
    else:
        initial_estimate = None

    solver = BQPSolver(problem, args.params, args.time, initial_estimate,
                       collect_heuristic_root_data=args.collect_root_data,
                       collect_sdp_bound_root_data=args.collect_root_data)

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
