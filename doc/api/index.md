# Python API

These pages document the project's Python modules. They are generated from
source signatures and docstrings, including the `.pyi` stub for the native
extension. Third-party code in `biqbin/external/` is not included.

- [Max-Cut and QUBO](solvers.md): problem, solver, and solution classes.
- [BQP](bqp.md): constrained binary quadratic problems and their parsers.
- [File parsers](parsers.md): reading problems and writing solutions.
- [CLI argument parsers](arguments.md): options used by the example commands.
- [Logging](logging.md): application and per-rank logging configuration.
- [Utilities](utilities.md): matrix conversion, validation, and serialization.
- [Native bindings](native.md): callbacks and low-level solver operations.

Max-Cut and QUBO classes are also re-exported from `biqbin` (for example,
`from biqbin import MaxCutSolver`). BQP classes currently live in the
repository-root module `biqbin_bqp`.

## MPI lifecycle

Call `biqbin.init()` before using a solver. Launch solver programs with at least
three MPI processes. The package registers `finalize()` to run at interpreter
exit; importing the package does not initialize MPI.

::: biqbin
    options:
      members:
        - init
        - finalize
