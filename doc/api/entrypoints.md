# Entry points

## MPI lifecycle

Launch solver programs with at least three MPI processes. Call `init()` on
each rank before creating a solver, and use `get_rank()` to restrict result
handling to rank 0. The package registers `finalize()` at interpreter exit;
normal applications do not need to call it explicitly.

The Python API currently supports **one solve per process**. Call `compute()`
once on every rank. A second attempt, including through a new solver object or
a different solver class, prints a fatal error and aborts the MPI job with
error code 10. Start a fresh MPI job to solve another problem. You may construct
multiple solver objects before that solve; `compute()` uses the callbacks of
the object on which it is called.

::: biqbin.init

::: biqbin.get_rank

::: biqbin.finalize

## Logging

Use `configure_logging()` for application logging. Command-line programs can
use `configure_rank_logging()` after MPI initialization to create separate
log files for each rank.

::: biqbin.configure_logging

::: biqbin.logging.configure_rank_logging

## Command-line programs

Run these scripts from the repository root:

| Script | Purpose |
| --- | --- |
| `biqbin_maxcut.py` | Solve a Max-Cut instance. |
| `biqbin_qubo.py` | Solve a QUBO instance. |
| `biqbin_bqp.py` | Solve a linearly constrained BQP instance. |
| `biqbin_heuristic.py` | Run the QUBO example with a local simulated-annealing heuristic. |

For example:

```bash
mpiexec -n 3 python biqbin_maxcut.py tests/rudy/g05_60.0.json
mpiexec -n 3 python biqbin_qubo.py tests/qubos/40/kcluster40_025_10_1.json
```

Use `python biqbin_maxcut.py --help` (or the corresponding script) to inspect
its arguments. See [input formats](../PROBLEM_INSTANCE_FORMATS.md), the
[BQP guide](../BQP_INPUT_EXAMPLE.md), and the [heuristic guide](../HEURISTIC.md)
for usage details. Argument parser implementation classes are deliberately
excluded from the generated API.

## BQP imports

Import BQP classes such as `ProblemBQP`, `SolutionBQP` and `BQPSolver`
from `biqbin`. Import `ArgParserBQP` from `biqbin.argparsers` when needed.
The `biqbin_bqp.py` module is a command-line entry point, not a library import path.
