from biqbin.logging import logger, configure_logging

from biqbin.biqbin_base import (ProblemMaxCut,
                                SolutionMaxCut,
                                MaxCutSolver,
                                ProblemQubo,
                                SolutionQubo,
                                QUBOSolver,
                                get_rank)

from biqbin.data_parsers import (MaxCutSolutionToJson,
                                 MaxCutFromJson,
                                 MaxCutFromMatrixMarket,
                                 MaxCutFromEdgeWeights,
                                 QuboFromJson,
                                 QuboSolutionToJson,
                                 QuboFromMatrixMarket,
                                 QuboFromEdgeWeights,
                                 QuboFromQPLIB)


__all__ = [
    'ProblemMaxCut',
    'SolutionMaxCut',
    'MaxCutSolver',
    'ProblemQubo',
    'SolutionQubo',
    'QUBOSolver',
    'logger',
    'configure_logging',
    'get_rank',
    'init',
    'MaxCutSolutionToJson',
    'MaxCutFromJson',
    'MaxCutFromMatrixMarket',
    'MaxCutFromEdgeWeights',
    'QuboSolutionToJson',
    'QuboFromJson',
    'QuboFromMatrixMarket',
    'QuboFromEdgeWeights',
    'QuboFromQPLIB'
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
