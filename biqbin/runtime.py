"""Shared reader -> problem -> solver -> solution -> writer execution."""

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from biqbin import data_parsers as readers
from biqbin.biqbin_base import BQPSolver, MaxCutSolver, QUBOSolver
from biqbin.biqbin_module import (
    abort_mpi,
    all_ranks_ok,
    broadcast_config,
    get_rank,
    parameter_schema,
    validate_parameters,
)
from biqbin.configuration import Option, load_config, load_legacy_parameters, resolve_options
from biqbin.logging import configure_rank_logging, configure_verbosity
from biqbin.problem import ProblemBQP, ProblemMaxCut, ProblemQubo
from biqbin.solution import SolutionMaxCut

Config = dict[str, Any]
SolverFactory = Callable[..., MaxCutSolver]


@dataclass(frozen=True)
class Components:
    problem: type[ProblemMaxCut]
    solver: type[MaxCutSolver]
    writer: type[readers.ToFile]
    readers: dict[str, type[readers.FromFile]]
    default_format: str


# Explicit selection keeps configuration serializable and avoids importing arbitrary code.
COMPONENTS = {
    "maxcut": Components(
        problem=ProblemMaxCut,
        solver=MaxCutSolver,
        writer=readers.MaxCutSolutionToJson,
        readers={
            "json": readers.MaxCutFromJson,
            "mm": readers.MaxCutFromMatrixMarket,
            "ew": readers.MaxCutFromEdgeWeights,
        },
        default_format="json",
    ),
    "qubo": Components(
        problem=ProblemQubo,
        solver=QUBOSolver,
        writer=readers.QuboSolutionToJson,
        readers={
            "json": readers.QuboFromJson,
            "mm": readers.QuboFromMatrixMarket,
            "ew": readers.QuboFromEdgeWeights,
            "qplib": readers.QuboFromQPLIB,
        },
        default_format="json",
    ),
    "bqp": Components(
        problem=ProblemBQP,
        solver=BQPSolver,
        writer=readers.BQPToJson,
        readers={
            "bqp": readers.BQPFromBQPFile,
            "json": readers.BQPFromJson,
            "dense": readers.BQPFromQBODense,
            "sparse": readers.BQPFromQBOSparse,
        },
        default_format="bqp",
    ),
}


class Runtime:
    """Own the orchestration options, independently of native algorithm parameters."""

    @classmethod
    def config_options(cls) -> dict[str, Option]:
        return {
            "initial_solution": Option(
                str, help="Initial binary solution JSON file", aliases=("-s", "--solution")
            ),
            "collect_root_data": Option(
                bool,
                default=False,
                help="Collect root callback data",
                aliases=("-c", "--collect-root-data"),
            ),
            "verbose": Option(
                int,
                default=0,
                minimum=0,
                help="Verbosity: 0 warning, 1 info, 2 debug",
                aliases=("-v", "--verbose"),
            ),
            "log_dir": Option(
                str, help="Additional per-rank log directory", aliases=("--log-dir",)
            ),
        }


def component_options(
    kind: str,
    fmt: str | None = None,
    solver_factory: SolverFactory | None = None,
) -> dict[str, dict[str, Option]]:
    if kind not in COMPONENTS:
        raise ValueError(f"Unknown problem kind {kind!r}; choose from {tuple(COMPONENTS)}")
    components = COMPONENTS[kind]
    fmt = fmt or components.default_format
    if fmt not in components.readers:
        raise ValueError(f"Unknown {kind} format {fmt!r}; choose from {tuple(components.readers)}")
    factory_type = solver_factory.func if isinstance(solver_factory, partial) else solver_factory
    factory_type = factory_type or components.solver
    solver_options = getattr(factory_type, "config_options", components.solver.config_options)()
    if isinstance(solver_factory, partial):
        # A bound, declared option supplies a default; config/CLI values still win.
        solver_options = {
            name: replace(option, default=solver_factory.keywords[name], required=False)
            if name in solver_factory.keywords
            else option
            for name, option in solver_options.items()
        }
    return {
        "reader": components.readers[fmt].config_options(),
        "problem": components.problem.config_options(),
        "solver": solver_options,
        "writer": components.writer.config_options(),
        "run": Runtime.config_options(),
    }


def resolve_config(
    config: Config | None = None,
    *,
    config_file: str | Path | None = None,
    params_file: str | Path | None = None,
    overrides: Config | None = None,
    solver_factory: SolverFactory | None = None,
) -> Config:
    """Resolve defaults < TOML/API config < explicit overrides, without side effects."""
    if config_file is not None and params_file is not None:
        raise ValueError("Use --config or --params, not both")
    values = load_config(config_file) if config_file is not None else dict(config or {})
    if config_file is not None:
        base = Path(config_file).resolve().parent
        for section, key in [
            ("reader", "path"),
            ("writer", "path"),
            ("run", "initial_solution"),
            ("run", "log_dir"),
        ]:
            entries = values.get(section, {})
            if isinstance(entries, dict) and isinstance(entries.get(key), str) and entries[key]:
                entries[key] = str(base / entries[key])
    if params_file is not None:
        values["solver"] = {**load_legacy_parameters(params_file), **values.get("solver", {})}
    overrides = overrides or {}
    allowed = {"kind", "format", "reader", "problem", "solver", "writer", "run"}
    for source in (values, overrides):
        if not isinstance(source, dict):
            raise ValueError("Configuration must be a mapping")
        unknown = source.keys() - allowed
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    kind = overrides.get("kind", values.get("kind"))
    fmt = overrides.get("format", values.get("format"))
    definitions = component_options(kind, fmt, solver_factory)
    result = {"kind": kind, "format": fmt or COMPONENTS[kind].default_format}
    for section, options in definitions.items():
        result[section] = resolve_options(options, values.get(section), overrides.get(section))
    native_keys = parameter_schema().keys()
    native = validate_parameters(
        {key: value for key, value in result["solver"].items() if key in native_keys}
    )
    result["solver"].update(native)
    if result["run"]["log_dir"] == "":
        result["run"]["log_dir"] = Path(result["reader"]["path"]).stem + ".logs"
    return result


def _collective_stage(action: Callable[[], Any], name: str) -> Any:
    """Finish local setup/output on every rank before entering another collective."""
    error = None
    result = None
    try:
        result = action()
    except Exception as exc:
        error = f"{name}: {exc}"
        print(f"BiqBin rank {get_rank()}: {error}", file=sys.stderr, flush=True)
    if not all_ranks_ok(error is None):
        raise ValueError(error or f"{name} failed on another MPI rank")
    return result


def run(
    config: Config | None = None,
    *,
    solver_factory: SolverFactory | None = None,
    config_file: str | Path | None = None,
    params_file: str | Path | None = None,
    overrides: Config | None = None,
) -> SolutionMaxCut | None:
    """Run collectively on all ranks; return a solution only on rank 0.

    A custom factory accepts ``problem`` and the standard solver keyword arguments.
    Configuration files are read on rank 0. Problem construction remains per-rank.
    MPI is initialized once and finalized at interpreter exit.
    """
    from biqbin import init

    init()
    rank = get_rank()
    message = None
    if rank == 0:
        try:
            resolved = resolve_config(
                config,
                config_file=config_file,
                params_file=params_file,
                overrides=overrides,
                solver_factory=solver_factory,
            )
            message = json.dumps({"config": resolved})
        except Exception as exc:
            message = json.dumps({"error": str(exc)})
    envelope = json.loads(broadcast_config(message))
    if "error" in envelope:
        raise ValueError(envelope["error"])
    config = envelope["config"]
    run_options = config["run"]
    verbosity = run_options["verbose"]
    configure_verbosity(verbosity)
    configure_rank_logging(run_options["log_dir"], rank)
    components = COMPONENTS[config["kind"]]
    reader_type = components.readers[config["format"]]
    factory = solver_factory or components.solver
    if isinstance(factory, partial):
        # Declared bound options have already been resolved. Remove those bindings
        # so native aliases such as time_limit cannot override the resolved values.
        factory = partial(
            factory.func,
            *factory.args,
            **{
                name: value
                for name, value in factory.keywords.items()
                if name not in config["solver"]
            },
        )

    def prepare():
        reader_options = dict(config["reader"])
        path = reader_options.pop("path")
        problem = reader_type(
            path, optimize_input=config["problem"]["optimize"], **reader_options
        ).read()
        estimate = None
        if rank == 0 and run_options["initial_solution"]:
            with open(run_options["initial_solution"]) as stream:
                estimate = np.asarray(json.load(stream)["initial_estimate"])
        native_keys = parameter_schema().keys()
        native = {key: value for key, value in config["solver"].items() if key in native_keys}
        extra = {key: value for key, value in config["solver"].items() if key not in native_keys}
        return factory(
            problem,
            parameters=native,
            initial_estimate=estimate,
            collect_heuristic_root_data=run_options["collect_root_data"],
            collect_sdp_bound_root_data=run_options["collect_root_data"],
            **extra,
        )

    solver = _collective_stage(prepare, "Input/solver setup")
    try:
        solution = solver.compute()
    except Exception as exc:
        print(f"BiqBin rank {rank}: solver failed: {exc}", file=sys.stderr, flush=True)
        abort_mpi(1)
        raise

    def write_result():
        if rank != 0:
            return
        if solution is None:
            raise ValueError("Solver returned no solution on rank 0")
        solution.meta_data["runtime_config"] = config
        solution.print_computed_solution(verbosity)
        options = dict(config["writer"])
        path = options.pop("path") or config["reader"]["path"] + ".output"
        actual_path = components.writer(solution).write(path, **options)
        print(f"Result written to {actual_path}")

    _collective_stage(write_result, "Output")
    return solution
