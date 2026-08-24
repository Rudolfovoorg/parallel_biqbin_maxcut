import json
import numpy as np
import scipy as sp
from abc import ABC, abstractmethod
from glob import glob

from biqbin.biqbin_base import ProblemMaxCut, ProblemQubo, SolutionMaxCut, SolutionQubo
import biqbin.external.pyqplib as pyqplib
from biqbin.utils import from_sparse, convert_numpy_to_json_serializable


class FromFile(ABC):
    """Interface for all file loader classes, subclasses must implement a `read` method,
    that takes a filename and returns a ProblemMaxCut or it's subclass (i.e. ProblemQubo)
    """

    def __init__(self, filename: str, problem_name: str | None = None, optimize_input: bool = False) -> None:
        """
        Args:
            filename (str): Path to problem instance file.
            problem_name (str | None, optional): Name of the problem instance. Defaults to filename.
            optimize_input (bool, optional): Divide the MaxCut adjacency matrix values by their greatest common divisor. Defaults to False.
        """
        self.filename = filename
        self.problem_name = problem_name if problem_name is not None else filename
        self.optimize_input = optimize_input

    @abstractmethod
    def read(self) -> ProblemMaxCut:
        ...


class MaxCutFromJson(FromFile):
    """MaxCut data parser for a JSON file containing a 'maxcut' key and scipy sparse.coo_matrix as value.
    """

    def read(self) -> ProblemMaxCut:
        """Read MaxCut json file that contains the adjacency matrix in scipy sparse.coo_matrix format.

        Returns:
            ProblemMaxCut: instance for MaxCutSolver.
        """
        with open(self.filename, "r") as f:
            mc_data = json.load(f)

        adj_matrix = from_sparse(mc_data["maxcut"])
        return ProblemMaxCut(adj_matrix, self.problem_name, optimize_mc_adj_matrix=self.optimize_input)


class MaxCutFromMatrixMarket(FromFile):
    def read(self) -> ProblemMaxCut:
        """Read a Max-Cut instance from MatrixMarket format.

        Returns:
            ProblemMaxCut: Problem instance for MaxCutSolver.
        """
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.mmread.html
        adj_matrix = sp.io.mmread(self.filename)
        if sp.sparse.issparse(adj_matrix):
            adj_matrix = adj_matrix.toarray()
        return ProblemMaxCut(adj_matrix, self.problem_name, self.optimize_input)


class MaxCutFromEdgeWeights(FromFile):
    """MaxCut edge weight data parser in Stanford Gset format style https://web.stanford.edu/~yyye/yyye/Gset/
    """

    def read(self) -> ProblemMaxCut:
        """Read MaxCut edge weight file (Stanford GSet format: https://web.stanford.edu/~yyye/yyye/Gset/) 
        and return the MaxCutProblem instance.

        Returns:
            ProblemMaxCut: Problem instance for MaxCutSolver.
        """
        with open(self.filename, 'r') as f:
            # Read number of vertices and edges
            num_vertices, num_edges = map(int, f.readline().split())
            adj_matrix = np.zeros(
                (num_vertices, num_vertices), dtype=np.float64)

            edges = np.atleast_2d(np.loadtxt(f, max_rows=num_edges))
            if edges.shape != (num_edges, 3):
                raise ValueError(
                    f'Expected {num_edges} edge rows with 3 columns, '
                    f'got {edges.shape}'
                )

        i, j, w = self.edge_weights_checks(edges, num_vertices)
        adj_matrix[i, j] = w
        adj_matrix[j, i] = w

        return ProblemMaxCut(adj_matrix, self.problem_name, optimize_mc_adj_matrix=self.optimize_input)

    def edge_weights_checks(self, edges: np.ndarray, num_vertices: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Safety checks for the edges and weights

        Args:
            edges (np.ndarray): Row, column, weights
            num_vertices (int): Number of vertices in the graph

        Raises:
            IndexError: If indices are not integers between 1 and num_verts
            ValueError: If weigths are not integers

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: rows, columns, weights
        """
        vertices = edges[:, :2]

        bad_rows = np.where(
            ~np.all(np.isfinite(vertices), axis=1)
            | ~np.all(vertices == np.round(vertices), axis=1)
            | np.any(vertices < 1, axis=1)
            | np.any(vertices > num_vertices, axis=1)
        )[0]

        if bad_rows.size:
            raise IndexError(
                f'Invalid vertices in rows {[int(row) + 1 for row in bad_rows]}: '
                f'vertices must be integers between 1 and {num_vertices}'
            )
        
        i = vertices[:, 0].astype(int) - 1
        j = vertices[:, 1].astype(int) - 1

        w = edges[:, 2]
        
        bad_weights_mask = ~np.isfinite(w) | (w != np.round(w))
        if np.any(bad_weights_mask):
            bad_rows = np.where(bad_weights_mask)[0] + 1
            raise ValueError(
                f'{self.filename}: Row(s) {[int(row) for row in bad_rows]} '
                f'have invalid weight(s), all weights must be finite integers!'
            )
        return i, j, w


class QuboFromJson(FromFile):
    """QUBO data parser for scipy sparse coo matrix in JSON format. Json must have a 'qubo' key and a scipy sparse.coo_matrix as value.
    """

    def read(self) -> ProblemQubo:
        """Read from the given filename and return the ProblemQubo used by the QuboSolver.
.
        Returns:
            ProblemQubo: Qubo problem instance for QuboSolver.
        """
        with open(self.filename, "r") as f:
            qubo_data = json.load(f)

        qubo = from_sparse(qubo_data["qubo"])
        offset = qubo_data.get('offset', 0)
        minimization = qubo_data.get('is_minimization', True)

        return ProblemQubo(Q=qubo,
                           offset=offset,
                           problem_name=self.problem_name,
                           is_minimization=minimization,
                           optimize_input=self.optimize_input)


class QuboFromMatrixMarket(FromFile):
    """QUBO data parser for MatrixMarket file format. Supports sparse and dense matrix formats.
    """

    def read(self) -> ProblemQubo:
        """Read a QUBO instance from MatrixMarket format.

        Returns:
            ProblemQubo: Problem instance for QUBOSolver.
        """
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.mmread.html
        Q = sp.io.mmread(self.filename)
        if sp.sparse.issparse(Q):
            Q = Q.toarray()
        return ProblemQubo(Q=Q, offset=0.0, problem_name=self.problem_name, is_minimization=True, optimize_input=self.optimize_input)


class QuboFromEdgeWeights(MaxCutFromEdgeWeights):
    """Qubo data parser in Stanford Gset format style https://web.stanford.edu/~yyye/yyye/Gset/
    """

    def read(self) -> ProblemQubo:
        """Read Qubo edge weight file (Stanford GSet format: https://web.stanford.edu/~yyye/yyye/Gset/) and return the QuboProblem instance.

        Returns:
            ProblemQubo: Problem instance for QUBOSolver.
        """
        with open(self.filename, 'r') as f:
            # Read number of vertices and edges
            num_vertices, num_edges = map(int, f.readline().split())
            Q = np.zeros(
                (num_vertices, num_vertices), dtype=np.float64)

            edges = np.atleast_2d(np.loadtxt(f, max_rows=num_edges))
            if edges.shape != (num_edges, 3):
                raise ValueError(
                    f'Expected {num_edges} edge rows with 3 columns, '
                    f'got {edges.shape}'
                )
        i, j, w = self.edge_weights_checks(edges, num_vertices)
        Q[i, j] = w

        return ProblemQubo(Q=Q,
                           offset=0.0,
                           problem_name=self.problem_name,
                           is_minimization=False,
                           optimize_input=self.optimize_input)


class QuboFromQPLIB(FromFile):
    """File reader for QPLIB instances https://qplib.zib.de/, 
    only unconstrained binary problems are allowed.
    """

    def read(self) -> ProblemQubo:
        """Reads qplib input file and constructs a QUBO problem.

        Raises:
            ValueError: Only unconstrained problems are valid
            ValueError: Only binary problems are valid

        Returns:
            ProblemQubo: Qubo Problem class that can be passed into QuboSolver.
        """

        problem = pyqplib.read_problem(self.filename)
        objective = problem.obj
        cons_type = problem.description.cons_type
        obj_type = problem.description.obj_type
        sense = problem.obj.sense == pyqplib.types.Sense.MINIMIZE

        # Check if the problem fits the solver
        if cons_type != pyqplib.ProblemConsType.UNCONSTRAINED:
            raise ValueError("Biqbin can only handle unconstrained problems!")
        if problem.description.var_type != pyqplib.ProblemVarType.BINARY:
            raise ValueError("Problem is not binary!")

        def to_coo_matrix(mat):
            return sp.sparse.coo_matrix(
                (mat.subdiag_vals, (mat.subdiag_rows, mat.subdiag_cols)),
                shape=mat.shape
            )

        if obj_type in [pyqplib.ProblemObjType.CONVEX, pyqplib.ProblemObjType.GENERAL]:
            Q: np.ndarray = to_coo_matrix(problem.obj.mat).toarray() # pyright: ignore[reportAttributeAccessIssue]
        else:
            Q = np.zeros((problem.description.num_vars, problem.description.num_vars))

        # QPLIB definition is 1/2 Quadratic + Linear + Offset
        Q /= 2
        np.fill_diagonal(Q, objective.lin)  # pyright: ignore[reportAttributeAccessIssue]
        
        return ProblemQubo(Q=Q, offset=objective.offset, problem_name=self.problem_name, is_minimization=sense, optimize_input=self.optimize_input)  # pyright: ignore[reportAttributeAccessIssue]



class ToFile(ABC):
    """Base abstract class for saving the solution to disk. All subclasses must implenent the
    `write` method that takes a solution and saves it as file.
    """

    @abstractmethod
    def write(self, filename: str, overwrite: bool = False, with_metadata: bool = True) -> None:
        ...

    @classmethod
    def get_output_path(cls, out_file: str, overwrite: bool) -> str:
        """Get the proper output path in case it already exists and we do not wish to overwrite.
        Attaches _N where N is the number of the next free output file. Adds .json if not already in the
        out_file's name.

        Args:
            out_file (str): output file path.
            overwrite (bool): if overwriting the outfile will not be changed.

        Returns:
            str: output file path
        """
        out_file = out_file[:-5] if out_file.endswith('.json') else out_file
        if overwrite:
            return out_file + '.json'

        file_count = len(glob(f'{out_file}*.json'))
        if file_count > 0:
            out_file += f'_{file_count}'
        return out_file + '.json'


class MaxCutSolutionToJson(ToFile):
    """Helper class to save the SolutionMaxCut as a json file.
    """

    def __init__(self, solution: SolutionMaxCut) -> None:
        self.solution: SolutionMaxCut = solution

    def write(self, filename: str, overwrite: bool = False, with_metadata: bool = True) -> None:
        """Save the solution as JSON file.

        Args:
            solution (SolutionMaxCut): Solution class returned by Biqbin after solving the problem
            with_metadata (bool, optional): Add meta_data to output file. Defaults to True.
        """

        # Check if output filename exists if we are not overriding and replace with filename_N.json
        output_path = self.get_output_path(filename, overwrite)

        save_output = {
            'maxcut': self.solution.solution
        }
        if with_metadata:
            save_output['meta_data'] = self.solution.meta_data

        with open(output_path, 'w') as f:
            json.dump(save_output, f,
                      default=convert_numpy_to_json_serializable)


class QuboSolutionToJson(ToFile):
    """Helper class to save QUBO solution as json file.
    """

    def __init__(self, solution: SolutionQubo) -> None:
        self.solution: SolutionQubo = solution

    def write(self, filename: str, overwrite: bool = False, with_metadata: bool = True, with_maxcut_solution: bool = False) -> None:
        """Save qubo solution to a json file.

        Args:
            solution (SolutionQubo): Solution class returned by Biqbin after solving the problem
            with_metadata (bool, optional): Add meta_data to output. Defaults to True.
            with_maxcut_solution (bool, optional): Add MaxCut solution to save output. Defaults to False.
        """

        # Check if output filename exists if we are not overriding and replace with filename_N.json
        output_path = self.get_output_path(filename, overwrite)

        save_output = {
            'qubo': self.solution.solution
        }

        if with_maxcut_solution:
            save_output['maxcut'] = self.solution.solution_maxcut
        if with_metadata:
            save_output['meta_data'] = self.solution.meta_data

        with open(output_path, 'w') as f:
            json.dump(save_output, f,
                      default=convert_numpy_to_json_serializable)
