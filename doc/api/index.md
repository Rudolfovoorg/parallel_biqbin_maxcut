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
