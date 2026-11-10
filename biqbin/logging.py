"""Shared Python/native logging, configured explicitly by CLI or application."""

import logging
import sys
from os import PathLike
from pathlib import Path


# Keep this name in sync with the native bridge in src/wrapper.cpp.
logger = logging.getLogger("biqbin")
logger.addHandler(logging.NullHandler())

DEFAULT_FORMAT = "%(asctime)s:%(levelname)s:%(name)s:rank=%(rank)s:%(message)s"
_configured_handlers: list[logging.Handler] = []
_mpi_rank: int | str = "-"


def _set_mpi_rank(rank: int) -> None:
    """Cache rank after MPI initialization; formatting never calls MPI."""
    global _mpi_rank
    _mpi_rank = rank


class _RankFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = _mpi_rank
        return True


def configure_logging(
    level: int | str = logging.WARNING,
    *,
    filename: str | PathLike[str] | None = None,
    format: str = DEFAULT_FORMAT,
) -> logging.Logger:
    """Configure stderr and an optional UTF-8 append-only log file.

    Args:
        level: Python logging severity number or standard level name.
        filename: File to append to, or None for stderr only.
        format: Percent-style logging format shared by both handlers. The
            rank field is "-" before MPI initialization, then the cached rank.

    Returns:
        The shared Python/native BiqBin logger.

    Raises:
        OSError: The requested file cannot be opened.
        ValueError: The level or format is invalid.

    Repeated calls replace and close only handlers created here. The root
    logger and application-owned handlers are left alone. Propagation is
    disabled after configuration to avoid duplicate output via root handlers.
    For MPI jobs, use a distinct filename per rank (no MPI calls are made here).
    Applications may instead configure ``logger`` using standard logging APIs.
    """
    # Validate configuration before replacing working handlers.
    formatter = logging.Formatter(format)
    console = logging.StreamHandler()
    console.setLevel(level)
    handlers: list[logging.Handler] = [console]
    if filename is not None:
        handlers.append(logging.FileHandler(filename, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(_RankFilter())

    for handler in _configured_handlers:
        logger.removeHandler(handler)
        handler.close()
    _configured_handlers[:] = handlers
    for handler in handlers:
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def configure_rank_logging(directory: str | PathLike[str] | None, rank: int) -> None:
    """Configure console and rank-file output after MPI initialization.

    Args:
        directory: Log directory, or None to retain console-only logging.
        rank: Nonnegative rank in the initialized MPI communicator.

    Existing rank files are appended to. Concurrent jobs must use separate
    directories. Directory creation or file-opening failures are reported to
    stderr and abort the MPI job with code 10, so peers cannot hang in solver
    communication. This function does not initialize MPI.
    """
    if rank < 0:
        raise ValueError("MPI rank must be nonnegative")
    _set_mpi_rank(rank)
    if directory is None:
        return
    path = Path(directory)
    try:
        path.mkdir(parents=True, exist_ok=True)
        configure_logging(logger.level, filename=path / f"rank-{rank:05d}.log")
    except OSError as error:
        from biqbin.biqbin_module import abort_mpi

        # Report independently of handlers: configuring those is what failed.
        try:
            print(f"BiqBin rank {rank}: cannot configure log directory {str(path)!r}: {error}",
                  file=sys.stderr, flush=True)
        finally:
            abort_mpi(10)
