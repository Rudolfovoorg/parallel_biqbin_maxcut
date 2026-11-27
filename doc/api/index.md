# Python API

This reference covers the main application API. It is organized by what you
work with, rather than by the files that implement it.

| Problem type | Problem | Solver | Result |
| --- | --- | --- | --- |
| Max-Cut | [ProblemMaxCut](problems.md#biqbin.ProblemMaxCut) | [MaxCutSolver](solvers.md#biqbin.MaxCutSolver) | [SolutionMaxCut](solutions.md#biqbin.SolutionMaxCut) |
| QUBO | [ProblemQubo](problems.md#biqbin.ProblemQubo) | [QUBOSolver](solvers.md#biqbin.QUBOSolver) | [SolutionQubo](solutions.md#biqbin.SolutionQubo) |
| BQP | [ProblemBQP](problems.md#biqbin.ProblemBQP) | [BQPSolver](solvers.md#biqbin.BQPSolver) | [SolutionBQP](solutions.md#biqbin.SolutionBQP) |

See [entry points](entrypoints.md) for MPI initialization, logging, and the
command-line programs. The [input format guide](../PROBLEM_INSTANCE_FORMATS.md)
and [custom solver guide](../CUSTOM_SOLVER.md) cover loading instances and
implementing callbacks. See [utilities](utilities.md) for matrix conversion,
QUBO export, JSON encoding, and result-data analysis.

## Runnable QUBO example

After [installing and building BiqBin](../INSTALLATION.md), run the
[example script](../examples/qubo_setup_example.py) from the repository root:

```bash
export OPENBLAS_NUM_THREADS=1 GOTO_NUM_THREADS=1 OMP_NUM_THREADS=1
mpiexec -n 3 python -m doc.examples.qubo_setup_example
```

Use module invocation (`-m`) so Python can find the local `biqbin` package.
The default `params` file is also resolved from the repository root.

The example minimizes `x.T @ Q @ x + 10` for binary `x`:

```python
import numpy as np
from biqbin import ProblemQubo, QUBOSolver, init

init()
problem = ProblemQubo(
    Q=np.array([[-1, 3], [3, -1]]),
    offset=10,
    problem_name="example_qubo",
    is_minimization=True,
)
result = QUBOSolver(problem, params="params").compute()
if result is not None:
    print("Objective:", result.solution["computed_val"])
    print("Binary solution:", result.solution["x"])
```

Every rank calls `init()` and `compute()`. Only rank 0 receives a result;
workers receive `None`. MPI finalization is registered automatically.
The optimal objective is **9**, with either `[1, 0]` or `[0, 1]` as a valid
optimal solution. `computed_val` includes the offset.

The script also writes `example_qubo.json` in the current directory on rank 0,
overwriting an existing file of that name. This is the **problem input**, not
the solution. Solve the exported input through the CLI with:

```bash
mpiexec -n 3 python biqbin_qubo.py example_qubo.json -o example_qubo.result.json
```

The CLI writes its result to `example_qubo.result.json`; if that path already
exists, it chooses a numbered filename. Add `--overwrite` to replace it.

To customize the solver, start from the separate
[SDP and heuristic override template](../CUSTOM_SDP_HEURISTIC.md#custom-solver-template).
The example above uses BiqBin's built-in callbacks.

## Public imports

```python
from biqbin import ProblemMaxCut, ProblemQubo, MaxCutSolver, QUBOSolver
from biqbin import SolutionMaxCut, SolutionQubo
from biqbin import ProblemBQP, BQPSolver, SolutionBQP
```

All three problem types use the public `biqbin` exports. Problem definitions
live in `biqbin/problem.py`, results in `biqbin/solution.py`, and solvers in
`biqbin/biqbin_base.py`. The root-level scripts are command-line entry points.

## Reference scope

The API pages explicitly select the classes, methods, and utility functions
intended for callers.
Formatting helpers, conversion internals, parser base classes, and the raw
native module are not listed as separate APIs. Public inherited methods are
shown on the main classes where they are used.

To document another public API, add an explicit object directive to the
appropriate page in `doc/api/`. Avoid whole-module directives, which expose
implementation helpers automatically.
