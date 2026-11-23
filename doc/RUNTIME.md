# Runtime pipeline

The shared runtime follows **reader → problem → solver → solution → writer**.
The three problem types use the same configuration handling and execution path.
Existing readers, problem classes, solver subclasses and writers remain usable
on their own through the Python API.

## Run from the command line

After [building BiqBin](INSTALLATION.md), run from the repository root with
at least three MPI processes:

```bash
export OPENBLAS_NUM_THREADS=1 GOTO_NUM_THREADS=1 OMP_NUM_THREADS=1
mpiexec -n 3 python -m biqbin qubo \
  tests/qubos/40/kcluster40_025_10_1.json -o result.json
```

Replace `qubo` with `maxcut` or `bqp` for the other problem types. Use
`python -m biqbin qubo --help` to inspect options. The original
`biqbin_maxcut.py`, `biqbin_qubo.py` and `biqbin_bqp.py` scripts are compatibility
entry points for the same runtime.

No `params` file in the working directory is required. Built-in solver defaults
match the values in the repository's complete `params` file. This changes the
old compiled fallback values for `init_bundle_iter` from 3 to 5 and
`violated_TriIneq` from 0.001 to 0.05; set them explicitly to retain those older
fallbacks when migrating a partial parameter file.

## Configuration files and overrides

A TOML file can configure a complete run. For example, save this as `run.toml`
next to an `instance.json` QUBO input:

```toml
kind = "qubo"
format = "json"

[reader]
path = "instance.json"

[problem]
optimize = false

[solver]
time_limit = 600
max_bundle_iter = 20

[run]
collect_root_data = true
verbose = 1

[writer]
path = "result.json"
overwrite = false
with_metadata = true
with_maxcut_solution = true
```

Run it with:

```bash
mpiexec -n 3 python -m biqbin qubo --config run.toml
```

Paths supplied in TOML are relative to that file. Paths supplied on the command
line are relative to the current working directory. Use absolute paths when
sharing a run configuration between directories or HPC jobs.

Precedence, from lowest to highest, is:

1. Component defaults.
2. The TOML configuration.
3. An explicitly selected legacy `-p/--params` file, for native solver parameters.
4. Explicit command-line overrides.

Omitted command-line flags do not reset configured values. For example:

```bash
mpiexec -n 3 python -m biqbin qubo --config run.toml \
  --param max_bundle_iter=25 --no-optimize --no-collect-root-data
```

Use `--param NAME=VALUE` for native solver parameters. Unknown names and invalid
values fail before solving. TOML `solver.time_limit` and the legacy parameter
file's `time_limit` use seconds; `0` means no solver time limit. An explicit
`-t 0` disables a configured limit.

Boolean overrides include `--no-optimize`, `--no-overwrite`,
`--no-collect-root-data`, `--no-metadata` and `--no-maxcut-solution`.

Without a configured output path, the result is written alongside the input
using the `.output.json` suffix. Existing results are preserved by selecting an
unused numbered filename, unless overwrite is explicitly enabled.

## Python API

The runner initializes MPI and performs all stages. Every rank must call it;
only rank 0 receives a solution. This example uses the same configuration as
the CLI:

```python
from biqbin.runtime import run

solution = run({
    "kind": "qubo",
    "reader": {"path": "instance.json"},
    "solver": {"time_limit": 600},
    "writer": {"path": "result.json"},
})
if solution is not None:
    print(solution.solution["computed_val"])
```

Run the script with `mpiexec -n 3 python your_script.py`, with the package
available to that Python environment. The source checkout currently supplies
the package; pip installation is a separate packaging task.

For in-memory problems, continue to construct `ProblemQubo`, `QUBOSolver` and
other public classes directly. Solver parameters can be supplied as a dictionary
using `parameters=...`; the solver passes validated values into the native
implementation. A filename is only needed when explicitly importing a legacy
parameter file.

## Custom solvers

Keep custom SDP and heuristic implementations in solver subclasses, as described
in the [callback guide](CUSTOM_SDP_HEURISTIC.md). Pass the subclass as the
runner's factory when it uses the standard constructor:

```python
from biqbin.runtime import run
from my_solver import MyQUBOSolver

run(
    {"kind": "qubo", "reader": {"path": "instance.json"}},
    solver_factory=MyQUBOSolver,
)
```

For additional constructor arguments, use a factory accepting
`factory(problem, **solver_kwargs)`, or `functools.partial`:

```python
from functools import partial

run(config, solver_factory=partial(MyQUBOSolver, sampler=my_sampler))
```

The factory is called on every MPI rank. The shared runtime supplies validated
native parameters, an initial estimate when applicable, and root-data collection
settings. Configuration files select built-in components by name; they do not
import arbitrary Python code.

## Component-owned options

Readers, problems, solvers and writers declare their configuration through the
`config_options()` class method. The shared resolver uses those definitions
for defaults and value validation. A component receives only its own settings;
CLI parsing and MPI lifecycle handling stay in the runtime layer.

Subclass definitions can extend inherited options. Native solver options are
based on the native parameter definitions, so the Python layer does not maintain
a second independent set of algorithm defaults.

## MPI and reproducibility

Rank 0 resolves the configuration and distributes it to the other ranks.
All ranks construct the problem and participate in the solver. Rank 0 prints
and writes the result. Setup failures are coordinated so one failing worker
does not leave the others waiting in the solver.

With metadata enabled, the result records `runtime_config` and
`solver_parameters`. The latter contains the native settings after adjustments
that depend on problem size, such as `adjust_TriIneq`.

On HPC systems, load the same compatible Python, MPI and BLAS environment used
for the native build, and launch inside a scheduler allocation using the site's
supported MPI launcher. See the [HPC guide](HPC_EXAMPLE.md).
