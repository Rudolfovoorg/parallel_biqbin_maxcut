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

_initialized = False

def init():
    """Initialize the MPI environment. Must be called before creating any solver."""
    global _initialized
    if not _initialized:
        from biqbin.biqbin_module import init_mpi
        init_mpi()
        _initialized = True
        