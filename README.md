# BiqBin: A Parallel Solver for Max-Cut and Binary Quadratic Problems

## Quantum solver for hard BInary Quadratic problems (QBIQ)

**ARIS Project (L1-60136), 1 January 2025 – 31 December 2027**

[ARIS project information](https://cris.cobiss.net/ecris/si/en/project/22627)

**Copyright © 2021 — The BiqBin Project**

Funded by **FWF (I 3199-N31)** and **ARRS (P2-0162)**.

This software implements the algorithm described in:

> Nicolò Gusmeroli, Timotej Hrga, Borut Lužar, Janez Povh, Melanie Siebenhofer, and Angelika Wiegele.
>
> **BiqBin: A Parallel Branch-and-Bound Solver for Binary Quadratic Problems with Linear Constraints**
>
> *ACM Transactions on Mathematical Software*, 48(2), Article 15, June 2022, 31 pages.
>
> [DOI: 10.1145/3514039](https://doi.org/10.1145/3514039) · [arXiv](https://arxiv.org/abs/2009.06240)

## Overview

BiqBin is a parallel branch-and-bound solver for Max-Cut and binary quadratic optimization problems.

The repository provides:

- A native C solver using MPI.
- Python bindings implemented with pybind11.
- Max-Cut, QUBO and general BQP problem support.
- Configurable SDP bounds and heuristic callbacks.
- Structured Python and native logging, including per-rank log files.
- Automated testing across Python and MPI implementations.
- Generated documentation and Python API references.

BiqBin requires **at least three MPI processes** for solver execution: one master and two workers.

## Quick start

From the repository root, follow the [installation guide](doc/INSTALLATION.md)
to install native dependencies and activate a Python environment. Then:

```bash
python -m pip install -r requirements.txt
make
export OPENBLAS_NUM_THREADS=1 GOTO_NUM_THREADS=1 OMP_NUM_THREADS=1
mpiexec -n 3 python biqbin_qubo.py \
  tests/qubos/40/kcluster40_025_10_1.json -o qubo_result.json
```

The result is written to `qubo_result.json`. If it already exists, the CLI
chooses a numbered filename; add `--overwrite` to replace it. Keep the same
Python environment active when building and running the solver.

For Max-Cut or constrained BQP, see [Running BiqBin](#running-biqbin).
For Python integration, start with the [runnable QUBO API example](doc/api/index.md#runnable-qubo-example).
For custom callbacks, use the separate [SDP and heuristic template](doc/CUSTOM_SDP_HEURISTIC.md#custom-solver-template).

## License

BiqBin is free software distributed under the **GNU General Public License, version 3 or later**.

You may use, modify and redistribute this software under the terms of the license.

The software is provided without warranty, including implied warranties of merchantability or fitness for a particular purpose.

See the [GNU GPL](https://www.gnu.org/licenses/) for details.

---

## Installation

BiqBin targets Linux environments.

For detailed installation instructions, see:

- [Installation guide](doc/INSTALLATION.md) — native/Conda and Docker environments.
- [HPC example](doc/HPC_EXAMPLE.md) — cluster-oriented installation and SLURM execution.

### Requirements

Native builds require:

- Python **3.12 or later**.
- GCC/G++ and GNU Make.
- Python development headers.
- An MPI implementation: **MPICH** or **OpenMPI**.
- OpenBLAS.
- The Python packages listed in `requirements.txt`.

In the Python environment created using the installation guide, install the runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

For development and testing:

```bash
python -m pip install -r requirements-dev.txt
```

### Build

From the repository root:

```bash
make
```

This builds the native executable and Python extension.

The default build is incremental: unchanged targets are reused, while missing or outdated targets are rebuilt.

To force a complete rebuild:

```bash
make rebuild
```

Or explicitly:

```bash
make clean
make
```

**Important:** GNU Make does not automatically detect changes to compilation variables such as `LOG_LEVEL`, the compiler or MPI implementation. Perform a clean rebuild when changing these settings.

---

## Docker

BiqBin uses a **multi-stage Dockerfile** with two user-facing targets.

| Target | Purpose |
|---|---|
| `runtime` | Standard environment containing the compiled solver and Python bindings |
| `ci` | Development and testing environment, including pytest, Ruff and coverage tooling |

Both targets share the same base environment and native build configuration.

The default environment uses Python 3.12, Debian Bookworm and MPICH.

### Standard runtime image

Build the default image:

```bash
docker build -t parallel-biqbin:1.0.0 .
```

Or use the Makefile:

```bash
make docker
```

The `runtime` target is the default Docker build target.

It runs as the nonroot user `biqbin`, with UID and GID `1000` by default.
The working directory is `/solver`; the bundled files, `/data`, and the user's
home directory `/home/biqbin` are writable by that account.

Open an interactive shell:

```bash
docker run --rm -it parallel-biqbin:1.0.0 bash
```

Alternatively:

```bash
make docker-shell
```

The Makefile shell target mounts the configured local `DATA_DIR` at `/data` inside the container.

### Development and CI image

Build the testing environment:

```bash
docker build --target ci -t parallel-biqbin-dev:1.0.0 .
```

Or:

```bash
make docker-dev
```

Run the complete regression suite inside Docker:

```bash
make docker-test
```

This builds the development image if needed and runs the tests inside it.

### Custom Python and MPI versions

The Dockerfile supports configurable Python and MPI implementations:

```bash
docker build \
  --target runtime \
  --build-arg PYTHON_VERSION=3.13 \
  --build-arg MPI_IMPL=mpich \
  -t biqbin-py313 .
```

For an OpenMPI development environment:

```bash
docker build \
  --target ci \
  --build-arg MPI_IMPL=openmpi \
  -t biqbin-openmpi-ci .
```

Supported `MPI_IMPL` values are `mpich` and `openmpi`.

The CI image permits OpenMPI execution as root and allows MPI oversubscription
for tests running on resource-limited containers. The standard runtime image
runs as a nonroot user and does not enable either setting.

When running actual performance benchmarks, allocate sufficient CPU resources rather than relying on oversubscription.

### Working with local problem instances

You can mount a local directory into the container:

```bash
mkdir -p instances
docker run --rm -it \
  -v "$PWD/instances:/data" \
  parallel-biqbin:1.0.0 bash
```

Problem files can then be accessed at `/data` from within the container.

A bind mount uses the host directory's ownership and permissions. On Linux,
you can set the runtime user's UID and GID at build time to match your account;
see [runtime user and host permissions](doc/INSTALLATION.md#runtime-user-and-host-permissions).
Write results and logs under the mounted `/data` directory to retain them after
the container exits. The native executable writes output beside its input file,
so that directory must be writable.

---

## Development and testing

The Makefile provides build, lint, testing, coverage and documentation targets.

### Running tests

Run the complete regression suite:

```bash
make test
```

The `tests` target is an alias:

```bash
make tests
```

The full suite includes native solver correctness, Python bindings, logging, summaries, input parsing, Max-Cut, QUBO, QPLIB, BQP, custom heuristics, initial solutions and MPI failure handling.

Tests are executed sequentially because some test groups generate and inspect shared output files. Avoid running separate test targets concurrently in the same checkout.

Plain `pytest` is **not equivalent** to `make test`: some tests require problem-instance arguments and outputs generated by preceding MPI runs.

Run a smaller subset:

```bash
make test-unit
```

Or specific groups:

```bash
make test-logging
make test-summaries
make test-parsers
make test-expected-failure
```

The default MPI solver timeout is 180 seconds, followed by a 10-second forced-termination grace period.

For slower environments:

```bash
make test MPI_TIMEOUT=600
```

MPI expected-failure tests have their own timeout handling to detect hangs.

### Linting and formatting

Python linting and formatting use [Ruff](https://docs.astral.sh/ruff/), configured in `pyproject.toml`.

Check lint rules:

```bash
make lint
```

Check formatting without modifying files:

```bash
make format-check
```

Format the Python source:

```bash
make format
```

Vendored third-party Python code is excluded from the project's Ruff checks.

### Solver behavior diagnostics

Output comparisons retain strict correctness checks and also write readable
expected-versus-actual reports to `artifacts/tests/behavior/<test-target>.txt`.
Each instance lists changed fields, expected and actual values, numeric deltas
(actual minus expected), and changed vector positions (1-based). Matching fields
are listed separately. Reports include solution vectors, root bounds and heuristic
values, node/call counts, runtimes, and custom-callback counters. Behavioral
differences alone do not fail tests; existing correctness assertions still apply.

The CI diagnostic artifact includes these reports under `tests/behavior/`
(the workflow copies the contents of the local `artifacts/` directory).
There is a separate report for each test target, so repeated instances in different
suites are retained. Reports are written before correctness checks and replaced
on the next run of that target. A direct pytest invocation defaults to
`artifacts/tests/behavior/output.txt`; use `--behavior-report=PATH` to choose another file.

### Python coverage

Collect coverage across the complete Makefile regression suite:

```bash
make coverage
```

This generates:

| Output | Description |
|---|---|
| `artifacts/coverage/.coverage` | Raw coverage database |
| `artifacts/coverage/coverage.xml` | XML report for automated tooling |
| `artifacts/coverage/htmlcov/index.html` | Annotated HTML coverage report |
| `artifacts/tests/junit/` | JUnit XML reports for pytest groups |

Open the HTML report locally:

```bash
xdg-open artifacts/coverage/htmlcov/index.html
```

Coverage includes Python statements and branches, excluding vendored code.

The Makefile instruments its MPI solver commands with `coverage run --parallel-mode`
and combines their per-process data with pytest coverage. Native C/C++ execution
is not measured. Other subprocesses launched directly by tests are not automatically
instrumented.
Coverage adds runtime overhead; use `make test` without coverage for timing comparisons.

Coverage is currently informational; the CI pipeline does not enforce a minimum percentage.

### Continuous integration

GitHub Actions runs automated checks when a pull request targeting any branch
is opened, reopened, or updated with new commits. Pushes to branches without an
open pull request do not run CI. Manual runs are also supported through
`workflow_dispatch`.

These checks provide results before merging. Merging or closing a pull request
does not start another run.

The workflow includes:

| Job | Environment | Checks |
|---|---|---|
| Ruff | Python 3.12 | Linting and formatting |
| Documentation | Python 3.12 | Strict Zensical build |
| Full regression suite | Python 3.12 + MPICH | Complete tests, Python coverage and JUnit |
| Python compatibility | Python 3.13 + MPICH | Focused unit tests |
| MPI compatibility | Python 3.12 + OpenMPI | Logging, Max-Cut and failure-handling smoke tests |
| Runtime images | Python 3.12 + MPICH/OpenMPI | Nonroot MPI solve and writable output paths |

The test environments are built from the Dockerfile's `ci` stage.

CI uses the normal INFO-level native logging build configuration.

Build and test output is captured in diagnostic artifacts, with one artifact per Python/MPI combination. Available coverage and JUnit reports are included.

The workflow validates documentation but **does not deploy a documentation website**, publish Docker images or create software releases.

For local checks before submitting a pull request:

```bash
make lint
make format-check
make docs
make test
```

For a container-based full regression run:

```bash
make docker-test
```

---

## Documentation

Documentation is maintained in `doc/` using Markdown.

[Zensical](https://zensical.org/) generates the HTML website using the existing `mkdocs.yml` configuration. Python API documentation uses `mkdocstrings`.

Install the documentation requirements:

```bash
python -m pip install -r requirements-docs.txt
```

Build the site:

```bash
make docs
```

Preview locally:

```bash
make docs-serve
```

Generated HTML is written to `site/`.

Documentation builds use strict validation: warnings cause the build to fail. Python API extraction does not require importing the native MPI extension.

See the [documentation home](doc/index.md) for the guides and API reference.

---

## Native logging

The `BIQBIN_LOG_*` macros in [`src/logging.h`](src/logging.h) provide printf-style logging from native C/C++ code.

For Python-integrated builds, native messages are forwarded to the shared `biqbin` Python logger. Pure-C builds write messages using the native logging fallback.

Importing BiqBin does not configure the root Python logger.

### Python logging configuration

Applications can configure stderr and optional UTF-8 file logging:

```python
from biqbin import configure_logging

configure_logging("DEBUG", filename="biqbin.log")
```

The default format includes a timestamp, severity, logger name and MPI rank. Before MPI initialization, the rank is displayed as `-`.

The configuration helper manages only handlers that it owns, leaving application-installed handlers intact.

### CLI logging and per-rank files

The command-line interface uses:

- Default verbosity: WARNING.
- `-v`: INFO messages.
- `-vv`: DEBUG messages.
- `--log-dir`: Write an additional file for each MPI rank.

Example:

```bash
mpiexec -n 3 python3 biqbin_maxcut.py \
  tests/rudy/g05_60.0.json \
  -v --log-dir
```

Without an explicit directory argument, the input filename stem determines the output directory, with a `.logs` suffix.

For example, `instance.json` produces:

```text
instance.logs/
├── rank-00000.log
├── rank-00001.log
└── rank-00002.log
```

Use an explicit directory to control the location:

```bash
mpiexec -n 3 python3 biqbin_maxcut.py \
  tests/rudy/g05_60.0.json \
  -v --log-dir ./logs
```

Log files are opened after MPI initialization and before problem loading. Existing files are appended to.

Failure to create the logging directory or open a rank's log file causes an error report and aborts the MPI job with code 10.

Use separate directories for concurrent jobs.

Custom command-line entry points can configure rank logging with `configure_rank_logging()` from `biqbin.logging` after argument parsing and MPI initialization.

### Compile-time logging level

Native logging supports compile-time severity filtering through `LOG_LEVEL`.

The default is INFO (`20`).

| Level | Value |
|---|---:|
| DEBUG | 10 |
| INFO | 20 |
| WARNING / WARN | 30 |
| ERROR | 40 |
| CRITICAL / FATAL | 50 |

Example:

```bash
make clean
make LOG_LEVEL=DEBUG
```

Level names are case-insensitive. Nonnegative decimal integer thresholds are also supported.

Messages below the compiled threshold are removed, including evaluation of their arguments.

Changing the compiled logging level requires rebuilding the native objects because Make does not automatically track changes to compilation variables.

Runtime Python logging configuration can filter messages that remain compiled into the binary. It cannot restore messages removed by compile-time filtering.

---

## Running BiqBin

BiqBin requires at least three MPI processes.

The Python API currently supports one solve per process. Start a fresh MPI job
for each problem; see [MPI lifecycle](doc/api/entrypoints.md#mpi-lifecycle).

For numerical libraries, prevent thread oversubscription:

```bash
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

### Native Max-Cut solver

Run the native executable:

```bash
mpiexec -n 3 ./biqbin_executable \
  tests/rudy/g05_60.0 params
```

General invocation:

```bash
mpiexec -n N ./biqbin_executable \
  problem_instance params
```

The native solver expects an edge-list problem instance and a solver parameter file.

### Python Max-Cut solver

```bash
mpiexec -n 3 python3 biqbin_maxcut.py \
  tests/rudy/g05_60.0.json
```

### Python QUBO solver

```bash
mpiexec -n 3 python3 biqbin_qubo.py \
  tests/qubos/40/kcluster40_025_10_1.json
```

### Custom QUBO heuristic

```bash
mpiexec -n 3 python3 biqbin_heuristic.py \
  tests/qubos/40/kcluster40_025_10_1.json
```

The `biqbin_heuristic.py` example demonstrates replacing the default heuristic with a custom sampler.

For custom SDP bounds and heuristics, see [Custom solver guide](doc/CUSTOM_SOLVER.md) and [Custom SDP/heuristic callbacks](doc/CUSTOM_SDP_HEURISTIC.md).

### General BQP solver

```bash
mpiexec -n 3 python3 biqbin_bqp.py \
  tests/bqp/test_bqp.data
```

See [BQP input example](doc/BQP_INPUT_EXAMPLE.md) for supported representations.

### Common Python CLI options

| Option | Purpose |
|---|---|
| `-p PARAMS` | Custom solver parameter file |
| `-s FILEPATH` | Initial binary solution in JSON format |
| `-w`, `--overwrite` | Overwrite an existing output |
| `-o OUTPUT`, `--output OUTPUT` | Specify output filename |
| `-c`, `--collect-root-data` | Collect root-node SDP and heuristic data |
| `-O`, `--optimize` | Divide the applicable Max-Cut matrix by its GCD |
| `-t TIME`, `--time TIME` | Set the running time limit |
| `-v`, `--verbose` | Increase logging verbosity |
| `--log-dir [DIR]` | Enable additional per-rank log files |
| `--format FORMAT` | Select an input format |
| `-h`, `--help` | Display CLI help |

Run the relevant Python entry point with `--help` for its exact supported options.

Time limits follow Slurm-style syntax: `MM`, `MM:SS`, `HH:MM:SS`,
`D-HH`, `D-HH:MM` or `D-HH:MM:SS`. For example, `-t 01:30` means
90 seconds; use `-t 01:30:00` for one hour and thirty minutes.

### Problem and solution summaries

Problem objects provide a common textual summary containing the problem name, optimization settings, GCD information and original problem data.

`problem.format_summary(verbose=True)` includes additional transformed-problem details.

Solution summaries include runtime, termination status, branch-and-bound nodes, heuristic calls, worker counts and objective information.

For QUBO and BQP, verbose output includes the underlying Max-Cut result. BQP summaries additionally describe transformation-related constants and penalties.

---

## Supported problem formats

Max-Cut and QUBO support:

- `json` — default JSON representation.
- `mm` — Matrix Market.
- `ew` — edge-weight list.

QUBO additionally supports:

- `qplib` — QPLIB QBN instances.

General BQP supports its own text and JSON representations.

All supported optimization inputs require integer-valued coefficients or weights.

For triangular QUBO matrices, off-diagonal coefficients must be even under the current QUBO-to-Max-Cut transformation.

See [Problem instance formats](doc/PROBLEM_INSTANCE_FORMATS.md) for complete examples.

The default maximum problem size is **1024 nodes**, controlled by `NMAX` in the native API header. Increasing it requires modifying the constant and rebuilding.

---

## Solver parameters

BiqBin's parameter file controls the bundle method, cutting-plane separation, branching and time limits.

| Parameter | Description |
|---|---|
| `init_bundle_iter` | Initial bundle-method iterations |
| `max_bundle_iter` | Maximum bundle-method iterations |
| `triag_iter` | Triangle inequality separation iterations |
| `pent_iter` | Pentagonal inequality separation iterations |
| `hept_iter` | Heptagonal inequality separation iterations |
| `max_outer_iter` | Maximum cutting-plane outer iterations |
| `extra_iter` | Additional refinement iterations |
| `violated_TriIneq` | Triangle inequality violation threshold |
| `TriIneq` | Maximum triangle inequalities to add |
| `adjust_TriIneq` | Dynamically adjust triangle inequalities |
| `PentIneq` | Number of pentagonal inequalities |
| `HeptaIneq` | Number of heptagonal inequalities |
| `Pent_Trials` | Pentagonal separation trials |
| `Hepta_Trials` | Heptagonal separation trials |
| `include_Pent` | Include pentagonal inequalities |
| `include_Hepta` | Include heptagonal inequalities |
| `root` | Compute only the root SDP bound |
| `use_diff` | Add cutting planes only when needed |
| `time_limit` | Maximum runtime in seconds; `0` means no limit |
| `branchingStrategy` | `0`: least fractional; `1`: most fractional |

---

## Examples and further reading

Python entry points:

- [`biqbin_maxcut.py`](biqbin_maxcut.py) — Max-Cut.
- [`biqbin_qubo.py`](biqbin_qubo.py) — QUBO.
- [`biqbin_heuristic.py`](biqbin_heuristic.py) — custom heuristic.
- [`biqbin_bqp.py`](biqbin_bqp.py) — general BQP.

Documentation:

- [Installation](doc/INSTALLATION.md)
- [HPC execution](doc/HPC_EXAMPLE.md)
- [Problem formats](doc/PROBLEM_INSTANCE_FORMATS.md)
- [Custom solvers](doc/CUSTOM_SOLVER.md)
- [Custom SDP bounds and heuristics](doc/CUSTOM_SDP_HEURISTIC.md)
- [Python API](doc/api/index.md)

## Contact

For project inquiries:

Beno Zupanc — [GitHub](https://github.com/Zvmcevap) · [Email](mailto:beno.zupanc@rudolfovo.eu)