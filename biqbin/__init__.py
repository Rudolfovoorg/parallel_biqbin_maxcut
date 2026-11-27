from biqbin.biqbin_base import BQPSolver, MaxCutSolver, QUBOSolver, get_rank
from biqbin.data_parsers import (
    BQPFromBQPFile,
    BQPFromJson,
    BQPFromQBODense,
    BQPFromQBOSparse,
    BQPToJson,
    MaxCutFromEdgeWeights,
    MaxCutFromJson,
    MaxCutFromMatrixMarket,
    MaxCutSolutionToJson,
    QuboFromEdgeWeights,
    QuboFromJson,
    QuboFromMatrixMarket,
    QuboFromQPLIB,
    QuboSolutionToJson,
)
from biqbin.logging import configure_logging, logger
from biqbin.problem import ProblemBQP, ProblemMaxCut, ProblemQubo
from biqbin.solution import SolutionBQP, SolutionMaxCut, SolutionQubo

__all__ = [
    "ProblemBQP",
    "SolutionBQP",
    "BQPSolver",
    "BQPFromBQPFile",
    "BQPFromJson",
    "BQPFromQBODense",
    "BQPFromQBOSparse",
    "BQPToJson",
    "ProblemMaxCut",
    "SolutionMaxCut",
    "MaxCutSolver",
    "ProblemQubo",
    "SolutionQubo",
    "QUBOSolver",
    "logger",
    "configure_logging",
    "get_rank",
    "init",
    "MaxCutSolutionToJson",
    "MaxCutFromJson",
    "MaxCutFromMatrixMarket",
    "MaxCutFromEdgeWeights",
    "QuboSolutionToJson",
    "QuboFromJson",
    "QuboFromMatrixMarket",
    "QuboFromEdgeWeights",
    "QuboFromQPLIB",
]

import atexit

_initialized = False


def init():
    """Initialize the MPI environment. Must be called before creating any solver."""
    from biqbin.biqbin_module import init_mpi
    from biqbin.logging import _set_mpi_rank

    global _initialized
    if not _initialized:
        _, rank = init_mpi()
        _set_mpi_rank(rank)
        _initialized = True


def finalize():
    """Finalize MPI if initialized, is called atexit"""
    from biqbin.biqbin_module import finalize_mpi

    global _initialized

    if _initialized:
        try:
            finalize_mpi()
        except Exception as e:
            logger.warning(e)
            pass  # MPI may already be in a bad state
    _initialized = False


atexit.register(finalize)
