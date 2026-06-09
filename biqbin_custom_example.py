import numpy as np
import numpy.typing as npt

from biqbin import QUBOSolver
from biqbin.biqbin_module import BabNode, Problem

"""
    Custom SDP and Heuristic Solver examples
"""


class CustomSolver(QUBOSolver):
    """
    Has to inherit from ``MaxCutSolver`` class or a class derived from it (such as the ``QUBOSolver``).

    This is an example of a custom solver class that implements:
    - custom sdp bound on root node      (minimization lower bound)
    - custom heuristic on root node      (minimization upper bound)
    - custom sdp bound on non-root nodes (minimization lower bound)
    - custom heuristic on non-root nodes (minimization upper bound)

    Because both SDP and heuristic callbacks are overridden, this class does
    not need to call set_sdp_primal_solution().

    The example implementations return hardcoded values in place of an actual sdp or heuristic 
    and are meant for demonstration purposes only.
    """

    def root_sdp_bound(self, node: BabNode, P0: Problem, P: Problem, *args, **kwargs) -> float:
        """Custom SDP bound routine on the root node.

        By overriding this method you can use a custom SDP routine only on the root node.

        Args:
            node (BabNode): Current B&B node
            P0 (Problem): Original (full) problem
            P (Problem): Subproblem of the current node

        Returns:
            float: SDP value
        """
        # No call to set_sdp_primal_solution() is needed here because
        # root_heuristic is also overridden in this example.
        sdp_value: float = 10000 # some computed SDP value
        return sdp_value

    def root_heuristic(self, L: np.ndarray, *args, **kwargs) -> npt.NDArray[np.int32]:
        """Custom root node heuristic.

        By overriding the heuristic method you can use a custom heuristic only on the root node. 

        Returned solution binary vector needs to be of length L.shape[0] - 1
        """
        x: npt.ArrayLike = np.zeros(L.shape[0] - 1, dtype=np.int32)
        return x

    def sdp_bound(self, node: BabNode, P0: Problem, P: Problem, *args, **kwargs) -> float:
        """Custom SDP bound routine on non-root B&B nodes

        If ``root_sdp_bound`` is not overwritten, this will also be used on the root node.

        Args:
            node (BabNode): Current B&B node
            P0 (Problem): Original (full) problem
            P (Problem): Subproblem of the current node

        Returns:
            float: SDP solution value
        """

        # No call to set_sdp_primal_solution() is needed here only because
        # heuristic() is also overridden in this example.

        sdp_value: float = 10000
        return sdp_value

    def heuristic(self, L: np.ndarray, *args, **kwargs) -> npt.NDArray[np.int32]:
        """Custom heuristic on non-root B&B nodes. 

        If ``root_heuristic`` is not overwritten, this function will also be used on the root node.
        """

        x: npt.ArrayLike = np.zeros(L.shape[0] - 1, dtype=np.int32)
        return x


class CustomSDPSolver(QUBOSolver):
    """Example of a solver with only a custom SDP routine.

    Default Biqbin ``heuristic`` requires an SDP primal solution PSD matrix to be set. 
    The default ``sdp_routine`` does this internally, but should we use a custom sdp routine, 
    we need to set it manually with ``self.set_sdp_primal_solution``.

    This class implements the simplest example, using the identity matrix as the SDP
    primal solution and returning an arbitrary large value as the SDP value.
    """

    def sdp_bound(self, node: BabNode, P0: Problem, P: Problem, *args, **kwargs) -> float:
        """Custom SDP bound routine with the default Biqbin heuristic.

        Before returning we call ``set_sdp_primal_solution`` passing in a
        PSD matrix of the same size as subproblem Laplacean (P.L).
        """

        # SDP primal solution must be of shape (P.n, P.n) or P.L.shape
        X = np.identity(P.n)
        # Set the primal solution inside native Biqbin
        self.set_sdp_primal_solution(X)

        sdp_value: float = 10000 # some computed SDP value
        return sdp_value