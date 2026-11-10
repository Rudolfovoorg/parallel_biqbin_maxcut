# BiqBin: A Solver for Max-Cut and Unconstrained Binary Quadratic Problem

## **Quantum solver for hard BInary Quadratic problems (QBIQ)**

### **Aris Project (L1-60136), 1.1.2025 - 31.12.2027**

[Aris project link](https://cris.cobiss.net/ecris/si/en/project/22627)

---

**Copyright © 2021 — The BiqBin Project**  
Funded by **FWF (I 3199-N31)** and **ARRS (P2-0162)**

This program implements the algorithm presented in the publication:

> _Nicolò Gusmeroli, Timotej Hrga, Borut Lužar, Janez Povh, Melanie Siebenhofer, and Angelika Wiegele._  
> **"BiqBin: A Parallel Branch-and-Bound Solver for Binary Quadratic Problems with Linear Constraints"**  
> _ACM Trans. Math. Softw. 48, 2, Article 15 (June 2022), 31 pages._  
> [https://doi.org/10.1145/3514039](https://doi.org/10.1145/3514039)  
> [Also available on arXiv](https://arxiv.org/abs/2009.06240)

---

## License

This program is **Free Software**, released under the terms of the **GNU General Public License (GPL) v3**  
or (at your option) **any later version**.

You are free to:

- Use
- Modify
- Distribute

**without any warranty**; without even the implied warranty of **MERCHANTABILITY** or **FITNESS FOR A PARTICULAR PURPOSE**.

For more details, refer to the [GNU General Public License](https://www.gnu.org/licenses/).

---

## Setup

> **Note:** This readme is for PC example, for HPC please refer to [HPC_EXAMPLE.md](doc/HPC_EXAMPLE.md)

#### Biqbin requires **Linux operating system**.

It also requires:

- **build-essential** package for GCC, G++, make 
- Python version **3.12** or later
- Python development headers (the `python3-dev` package).
- MPI implementation (`OpenMPI` or `MPICH`)
- `OpenBLAS` mathematics library

The rest can be installed with `pip install -r requirements.txt` for [minimal requirements](requirements.txt) or `pip install -r requirements-dev.txt` for the [full requirements (including tests)](requirements-dev.txt):

```
# python>=3.12

numpy
pybind11
scipy
```

## [Installation readme](doc/INSTALLATION.md)

There are detailed descriptions on two seperate setup instruction, either [**Anaconda**](doc/INSTALLATION.md#conda-based-build) or [**Docker**](doc/INSTALLATION.md#docker) based builds.

### Native logging

The `BIQBIN_LOG_*` macros in [`src/logging.h`](src/logging.h) send printf-style messages through the existing Python logger `biqbin`, using its normal `logger.log()` method. The shared logger and configuration helper live in `biqbin/logging.py`. CLI verbosity configures stderr output; importing BiqBin does not configure the root logger. Add `__FILE__` or `__LINE__` to your message arguments when needed. Builds with `PURE_C` print these messages with `printf` instead.

CLI logs go to the terminal by default. Add `--log-dir` to also append logs to one file per MPI rank. With no directory argument, the input filename stem supplies the directory name in the current working directory (`instance.json` becomes `instance.logs`). Use `--log-dir PATH` for a custom directory; `-v`/`-vv` controls verbosity for both destinations:

```bash
mpirun -n 3 python biqbin_maxcut.py instance.json -v --log-dir
```

This creates `instance.logs/rank-00000.log`, `rank-00001.log`, and `rank-00002.log`. Files open after MPI initialization and before reading the problem. If a rank cannot create the directory or open its file, it reports the error to stderr and aborts the MPI job with code 10. Existing files are appended to; use different directories for concurrent jobs. Custom CLI entry points should call `configure_rank_logging(args.log_dir, get_rank())` from `biqbin.logging` after parsing arguments and initializing MPI.

Applications can explicitly configure console output and an optional UTF-8 log file:

```python
from biqbin import configure_logging

configure_logging("DEBUG", filename="biqbin.log")
```

The default format includes a local timestamp with milliseconds, the level, logger name, and MPI rank (`rank=-` before MPI initialization). For MPI runs, use a separate filename per rank, for example `filename=f"biqbin.{get_rank()}.log"` after MPI initialization. Reconfiguring replaces only handlers created by this helper; application-owned handlers remain intact. Configuration disables propagation to the root logger to avoid duplicate console output. For custom handler setups, use `biqbin.logger` directly. Pass `format="%(levelname)s:%(message)s"` to customize formatting, or level `5` for TRACE. Changing verbosity only changes diagnostics; it does not run additional heuristics.

Set the minimum compiled log level with `LOG_LEVEL`, which passes the corresponding numeric `-DBIQBIN_LOG_MIN_LEVEL` to both C and C++ compilers. Level names are case-insensitive:

```bash
make LOG_LEVEL=debug
```

`make LOG_LEVEL=DEBUG` and `make LOG_LEVEL=10` are equivalent. The default is INFO (`20`). Available names are TRACE (`5`, a custom level), DEBUG (`10`), INFO (`20`), WARNING (`30`, also WARN), ERROR (`40`), and CRITICAL (`50`, also FATAL). Nonnegative decimal integer thresholds are also supported (`0` is valid). Leading-zero values such as `010` or `08`, hexadecimal values, negative values, and fractions are rejected. Calls below the compiled threshold are removed, including evaluation of their arguments. Python's logger and handler levels still filter the remaining messages: use `-v` for INFO or `-vv` for DEBUG at runtime. TRACE also requires a Python logger level of `5`.

Rebuild all C and C++ objects after changing `LOG_LEVEL`; make does not track compiler flag changes. The default `make` target already performs a clean rebuild. This setting applies to `BIQBIN_LOG_*` calls; existing direct `printf` calls keep their current behavior.

## Usage

Problem and solution summaries start with their class name on the first line. Their shared `PrettyPrint` formatter keeps that heading beside the log prefix when using `logger.info("%s", problem)` or `logger.info("%s", solution)`, with all fields on subsequent lines.

Problem objects also share a summary: `str(problem)` prints the common name/optimization/GCD header and the original problem’s dimensions, coefficients, and applicable objective settings. `problem.format_summary(verbose=True)` additionally includes the converted Max-Cut problem and BQP transformation details.

Max-Cut, QUBO, and BQP print the same run statistics at every verbosity: problem name, runtime, time-limit status, B&B nodes, heuristic calls, workers used, input optimization, and GCD. Each summary then shows the original problem’s result. For QUBO and BQP, `-v` or `-vv` also shows the underlying Max-Cut result; BQP adds its rho, conversion constant, and penalty. In Python, `str(solution)` and `solution.print_computed_solution()` use the same summary; `solution.format_summary(verbose=True)` returns the detailed version as text.

> **NOTE:** Biqbin requires integer-valued Max-Cut edge weights and
> integer-valued QUBO/BQP coefficients.

> **NOTE:** Under the current QUBO-to-Max-Cut transformation, triangular QUBO
> matrices must have even off-diagonal coefficients so that the resulting
> Max-Cut weights remain integer-valued.

> **Min Processes:** Biqbin requires at least **3 mpi processes to run**!

> **Over Concurrency:** Depending on your system you must set `OpenBlas` environment variables, to prevent over threading which can **significantly** slow down your system:

```bash
 export OPENBLAS_NUM_THREADS=1
 export GOTO_NUM_THREADS=1
 export OMP_NUM_THREADS=1
```

### Original Biqbin Maxcut Parallel solver - C only

Example:

```bash
mpirun -n 3 ./biqbin tests/rudy/g05_60.0 params
```

General command:

```bash
mpirun -n N ./biqbin problem_instance params
```

- `num_processes`: number of processes to run the program using MPI, program needs at least 3 (1 master, and 2 worker process) to be used.
- `problem_instance`: A file containing the graph (in edge list format).
- `params`: The parameter file used to configure the solver.

---

## Python versions

### Python Wrapper for Biqbin Maxcut Parallel solver

Example:

```bash
mpirun -n 3 python3 biqbin_maxcut.py tests/rudy/g05_60.0.json
```

General command:

```bash
mpirun -n N python3 biqbin_maxcut.py problem_instance [-optional]
```

- `N`: number of processes to run the program using MPI, program needs at least 3 (1 master, and 2 worker process) to be used.
- `problem_instance`: Path to the problem instance file.
- `-p PARAMS`: Optional custom parameter file used to configure the solver, defaults to 'params'.
- `-s FILEPATH`: Optional filepath to an initial estimate solution. `JSON` file with `initial_estimate` key and a list of binary values as value.
- `-w`, `--overwrite`: Optional command to overwrite the output file if one already exists instead of appending '\_NUMBER'.
- `-o OUTPUT`, `--output OUTPUT`: Optional custom OUTPUT file name.
- `-c`, `--collect-root-data`: Collect heuristic and sdp data on root node (time taken and value returned)
- `-t TIME`, `--time TIME`: Set running time limit; acceptable time formats include "minutes", "minutes:seconds" "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
- `-O`, `--optimize`:  Divides the Max-Cut adjacency matrix by its GCD.
- `-v`, `--verbose`: Increases logging level from WARNING to -v for INFO or -vv for DEBUG
- `--log-dir [LOG_DIR]`: Also append logs to `rank-NNNNN.log` files in the supplied directory, or `<input stem>.logs` in the current directory when no path is supplied. Omit for terminal-only logging.
- `--format FORMAT`: Max-Cut problem instance file format. Valid formats are [('json', 'mm', 'ew')](doc/PROBLEM_INSTANCE_FORMATS.md), defaults to json.
- `-h`, `--help`: Show help message and exit.

---

### Python Wrapper for QUBO-s

Example:

```bash
mpirun -n 3 python3 biqbin_qubo.py tests/qubos/40/kcluster40_025_10_1.json
```

General command:

```bash
mpirun -n N python3 biqbin_qubo.py problem_instance [-optional]
```

- `N`: number of processes to run the program using MPI, program needs at least 3 (1 master, and 2 worker process) to be used.
- `problem_instance`: Path to the problem instance file.
- `-p PARAMS`: Optional custom parameter file used to configure the solver, defaults to 'params'.
- `-s FILEPATH`: Optional filepath to an initial estimate solution. `JSON` file with `initial_estimate` key and a list of binary values as value.
- `-w`, `--overwrite`: Optional command to overwrite the output file if one already exists instead of appending '\_NUMBER'.
- `-o OUTPUT`, `--output OUTPUT`: Optional custom OUTPUT file name.
- `-c`, `--collect-root-data`: Collect heuristic and sdp data on root node (time taken and value returned)
- `-O`, `--optimize`:  Divides the transformed Max-Cut adjacency matrix by its GCD.
- `-t TIME`, `--time TIME`: Set running time limit; acceptable time formats include "minutes", "minutes:seconds" "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
- `-v`, `--verbose`: Increases logging level from WARNING to -v for INFO or -vv for DEBUG
- `--log-dir [LOG_DIR]`: Also append logs to `rank-NNNNN.log` files in the supplied directory, or `<input stem>.logs` in the current directory when no path is supplied. Omit for terminal-only logging.
- `--format FORMAT`: QUBO problem instance file format. Valid formats are [('json', 'mm', 'ew', 'qplib')](doc/PROBLEM_INSTANCE_FORMATS.md), defaults to json.
- `-h`, `--help`: Show help message and exit.

---

### QUBO DWaveSampler

In `biqbin_heuristic.py` there is an example of a `QuboDwaveSampler` class which inhearits from `QUBOSolver` which overrides
Biqbins default heuristic. It takes an extra `sampler` argument plus any arguments Dwaves sampler takes.
In our example we use DWaves `neal.SimulatedAnnealingSampler`.

Example:

```bash
mpirun -n 3 python3 biqbin_heuristic.py tests/qubos/40/kcluster40_025_10_1.json
```

General command:

```bash
mpirun -n N python3 biqbin_heuristic.py problem_instance [-optional]
```

- `N`: Number of processes to run the program using MPI, program needs at least 3 (1 master, and 2 worker process) to be used.
- `problem_instance`: Path to the problem instance file.
- `-p PARAMS`: Optional custom parameter file used to configure the solver, defaults to 'params'.
- `-s FILEPATH`: Optional filepath to an initial estimate solution. `JSON` file with `initial_estimate` key and a list of binary [0, 1] values as value.
- `-w`, `--overwrite`: Optional command to overwrite the output file if one already exists instead of appending '\_NUMBER'.
- `-o OUTPUT`, `--output OUTPUT`: Optional custom OUTPUT file name.
- `-c`, `--collect-root-data`: Collect heuristic and sdp data on root node (time taken and value returned)
- `-O`, `--optimize`: Divides the transformed Max-Cut adjacency matrix by its GCD.
- `-t TIME`, `--time TIME`: Set running time limit; acceptable time formats include "minutes", "minutes:seconds" "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
- `-v`, `--verbose`: Increases logging level from WARNING to -v for INFO or -vv for DEBUG
- `--log-dir [LOG_DIR]`: Also append logs to `rank-NNNNN.log` files in the supplied directory, or `<input stem>.logs` in the current directory when no path is supplied. Omit for terminal-only logging.
- `--format FORMAT`: QUBO problem instance file format. Valid formats are [('json', 'mm', 'ew', 'qplib')](doc/PROBLEM_INSTANCE_FORMATS.md), defaults to 'json'.
- `-h`, `--help`: Show help message and exit.


### Python Wrapper for general BQP

Example:

```bash
mpirun -n 3 python3 biqbin_bqp.py tests/bqp/test_bqp.data
```

General command:

```bash
mpirun -n N python3 biqbin_bqp.py problem_instance [-optional]
```

- `N`: number of processes to run the program using MPI, program needs at least 3 (1 master, and 2 worker process) to be used.
- `problem_instance`: Path to the problem instance file.
- `-p PARAMS`: Optional custom parameter file used to configure the solver, defaults to 'params'.
- `-s FILEPATH`: Optional filepath to an initial estimate solution. `JSON` file with `initial_estimate` key and a list of binary [0, 1] values as value.
- `-w`, `--overwrite`: Optional command to overwrite the output file if one already exists instead of appending '\_NUMBER'.
- `-o OUTPUT`, `--output OUTPUT`: Optional custom OUTPUT file name.
- `-c`, `--collect-root-data`: Collect heuristic and sdp data on root node (time taken and value returned)
- `-O`, `--optimize`: Divides the transformed Max-Cut adjacency matrix by its GCD.
- `-t TIME`, `--time TIME`: Set running time limit; acceptable time formats include "minutes", "minutes:seconds" "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
- `-v`, `--verbose`: Increases logging level from WARNING to -v for INFO or -vv for DEBUG
- `--log-dir [LOG_DIR]`: Also append logs to `rank-NNNNN.log` files in the supplied directory, or `<input stem>.logs` in the current directory when no path is supplied. Omit for terminal-only logging.
- `--format FORMAT`: BQP problem instance file format. Valid formats are [('bqp', 'json')](doc/BQP_INPUT_EXAMPLE.md), defaults to 'bqp'.
- `-h`, `--help`: Show help message and exit.

Detailed explanation is available in [doc/BQP_INPUT_EXAMPLE.md](doc/BQP_INPUT_EXAMPLE.md).

---

### Examples

Please check the following Python files to find how to setup biqbin solver through Python

- [`biqbin_maxcut.py`](biqbin_maxcut.py): Example on how to run the default version of biqbin.
- [`biqbin_qubo.py`](biqbin_qubo.py): Example on how to run QUBO problem.
- [`biqbin_heuristic.py`](biqbin_heuristic.py): Example on how to use custom heuristic for lower bound estimation.
- [`biqbin_bqp.py`](biqbin_bqp.py): Example on how to run general BQP biqbin.

---

> **NOTE:** The default **maximum problem size** is **1024 nodes**.  
> If you require more, **edit** the value of `NMAX` in `biqbin_cpp_api.h`.

---

## Problem Instance Input files

> **NOTE:** Biqbin requires **integer**-valued Max-Cut edge weights and **integer**-valued QUBO/BQP coefficients.

Different Biqbin solvers support different file formats, [detailed explanation can be found here](doc/PROBLEM_INSTANCE_FORMATS.md).

Python versions can use the `--format` optional parameter to choose one of the following file formats
- `json` for the [biqbin json format](doc/PROBLEM_INSTANCE_FORMATS.md#json-input-example). Biqbin defaults to this.
- `mm` for the [MatrixMarket format](doc/PROBLEM_INSTANCE_FORMATS.md#matrix-market-format).
- `ew` for the [Edge-Weight list format](doc/PROBLEM_INSTANCE_FORMATS.md#edge-weight-list-input-example).

For QUBOs we also support:
- `qplib` for the [QPLIB file format](doc/PROBLEM_INSTANCE_FORMATS.md#qplib-format-for-qubos).

## Explanation of Parameters

| Parameter           | Description                                                                    |
| ------------------- | ------------------------------------------------------------------------------ |
| `init_bundle_iter`  | Initial number of iterations for the **bundle method**                         |
| `max_bundle_iter`   | Maximum number of iterations for the bundle method                             |
| `triag_iter`        | Number of iterations for **triangle inequality separation**                    |
| `pent_iter`         | Number of iterations for **pentagonal inequality separation**                  |
| `hept_iter`         | Number of iterations for **heptagonal inequality separation**                  |
| `max_outer_iter`    | Maximum number of **cutting plane algorithm** iterations                       |
| `extra_iter`        | Additional iterations for refinement or fallback in cutting plane algorithm    |
| `violated_TriIneq`  | Threshold for **triangle inequality violation**: `B(X) - 1 > violated_TriIneq` |
| `TriIneq`           | Maximum number of **triangle inequalities** added during separation            |
| `adjust_TriIneq`    | Whether to **adjust triangle inequalities dynamically** (0 or 1)               |
| `PentIneq`          | Number of **pentagonal inequalities** to add (usually `3 * Pent_Trials`)       |
| `HeptaIneq`         | Number of **heptagonal inequalities** to add (usually `4 * Hepta_Trials`)      |
| `Pent_Trials`       | Number of **simulated annealing trials** for pentagonal inequalities           |
| `Hepta_Trials`      | Number of **simulated annealing trials** for heptagonal inequalities           |
| `include_Pent`      | Include **pentagonal inequalities** in SDP bound (0 or 1)                      |
| `include_Hepta`     | Include **heptagonal inequalities** in SDP bound (0 or 1)                      |
| `root`              | If `1`, compute only **SDP bound at the root node**                            |
| `use_diff`          | If `1`, **only add cutting planes** when necessary to speed up B&B             |
| `time_limit`        | Maximum runtime in **seconds**. If `0`, runs until optimal solution is found   |
| `branchingStrategy` | Branching strategy:<br>`0 = LEAST_FRACTIONAL`<br>`1 = MOST_FRACTIONAL`         |

## Customizing the solver

Please refer to [`doc/CUSTOM_SOLVER.md`](doc/CUSTOM_SOLVER.md) on how to replace SDP bounds, root SDP bounds, heuristics, and root heuristics with your own custom routines.

## Contact information

For any inquiries about the software provided use the following contacts:

- Beno Zupanc: [Github account](https://github.com/Zvmcevap) or [email](beno.zupanc@rudolfovo.eu)

### Running tests

After building the native extension and installing the development dependencies:

```bash
make test-logging    # Logging, rank-file failures, and build-level validation
make test-summaries  # Problem/solution formatting and verbose sections
make test           # All test suites, including both targets above
```

`make tests` is an alias for `make test`. The full suite also includes standalone C,
Python solver, QPLIB, heuristic, initial-solution, parser, custom-callback, and MPI
failure tests. Run from an environment with MPI and the development dependencies
available (for example, `biqbin-py312`).
