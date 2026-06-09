from biqbin.biqbin_base import (ProblemMaxCut,
                                SolutionMaxCut,
                                MaxCutSolver,
                                ProblemQubo,
                                SolutionQubo,
                                QUBOSolver,
                                logger,
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
    global _initialized
    if not _initialized:
        init_mpi()
        _initialized = True


def finalize():
    """Finalize MPI if initialized, is called atexit"""
    from biqbin.biqbin_module import finalize_mpi
    global _initialized

    if _initialized:
        finalize_mpi()
    _initialized = False


atexit.register(finalize)
