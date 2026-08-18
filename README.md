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

- **build-essentials** package for GCC, G++, make 
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

## Usage

> **NOTE:** Biqbin can only solve problem instances with **integer edge weight**!  
> This applies both to the Maxcut and Qubo solvers.

> **NOTE**: QUBO instances must transform to integer-valued Max-Cut edge weights.
> For triangular QUBO matrices, off-diagonal coefficients must therefore be even.

> **Min Processes:** Biqbin requires needs at least **3 mpi processes to run**!

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
- `problem_instance`: A `JSON` serializable sparse coo adjacency matrix.
- `-p PARAMS`: Optional custom parameter file used to configure the solver, defaults to 'params'.
- `-s FILEPATH`: Optional filepath to an initial estimate solution. `JSON` file with `initial_estimate` key and a list of binary values as value.
- `-w`, `--overwrite`: Optional command to overwrite the output file if one already exists instead of appending '\_NUMBER'.
- `-o OUTPUT`, `--output OUTPUT`: Optional custom OUTPUT file name.
- `-e`, `--edge_weight`: Optional command to use edge weight `problem_instance` that C-only solver uses.
- `-t TIME`, `--time TIME`: Set running time limit; acceptable time formats include "minutes", "minutes:seconds" "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
- `-O`, `--optimize`: Divides the final input matrix values by their GCD
- `-v`, `--verbose`: Increases logging level from WARNING to -v for INFO or -vv for DEBUG
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
- `problem_instance`: `JSON` serializable dictionary containing "qubo" key and a sparse coo matrix for value (see tests/qubo/) folder for examples.
- `-p PARAMS`: Optional custom parameter file used to configure the solver, defaults to 'params'.
- `-s FILEPATH`: Optional filepath to an initial estimate solution. `JSON` file with `initial_estimate` key and a list of binary values as value.
- `-w`, `--overwrite`: Optional command to overwrite the output file if one already exists instead of appending '\_NUMBER'.
- `-o OUTPUT`, `--output OUTPUT`: Optional custom OUTPUT file name.
- `-O`, `--optimize`: Divide QUBO values by their GCD.
- `-t TIME`, `--time TIME`: Set running time limit; acceptable time formats include "minutes", "minutes:seconds" "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
- `-v`, `--verbose`: Increases logging level from WARNING to -v for INFO or -vv for DEBUG
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
- `problem_instance`: `JSON` serializable dictionary containing "qubo" key and a sparse coo matrix for value (see tests/qubo/) folder for examples.
- `-p PARAMS`: Optional custom parameter file used to configure the solver, defaults to 'params'.
- `-s FILEPATH`: Optional filepath to an initial estimate solution. `JSON` file with `initial_estimate` key and a list of binary values as value.
- `-w`, `--overwrite`: Optional command to overwrite the output file if one already exists instead of appending '\_NUMBER'.
- `-o OUTPUT`, `--output OUTPUT`: Optional custom OUTPUT file name.
- `-O`, `--optimize`: Divides the final input matrix values by their GCD
- `-d`, `--debug`: Enables debug logs
- `-i`, `--info`: Enable info logs
- `-t TIME`, `--time TIME`: Set running time limit; acceptable time formats include "minutes", "minutes:seconds" "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
- `-v`, `--verbose`: Increases logging level from WARNING to -v for INFO or -vv for DEBUG
- `--format FORMAT`: QUBO problem instance file format. Valid formats are [('json', 'mm', 'ew', 'qplib')](doc/PROBLEM_INSTANCE_FORMATS.md), defaults to 'json'.
- `-h`, `--help`: Show help message and exit.

### Python Wrapper for general BQP

> **NOTE:** BQP is considered to be a placeholder implementation at this time!

Example:

```bash
mpirun -n 3 python3 biqbin_bqp.py tests/bqp/test_bqp.data
```

General command:

```bash
mpirun -n N python3 biqbin_bqp.py problem_instance [-optional]
```

- `N`: number of processes to run the program using MPI, program needs at least 3 (1 master, and 2 worker process) to be used.
- `problem_instance`: .
- `-p PARAMS`: Optional custom parameter file used to configure the solver, defaults to 'params'.
- `-s FILEPATH`: Optional filepath to an initial estimate solution. `JSON` file with `initial_estimate` key and a list of binary values as value.
- `-w`, `--overwrite`: Optional command to overwrite the output file if one already exists instead of appending '\_NUMBER'.
- `-o OUTPUT`, `--output OUTPUT`: Optional custom OUTPUT file name.
- `-O`, `--optimize`: Divide QUBO values by their GCD.
- `-t TIME`, `--time TIME`: Set running time limit; acceptable time formats include "minutes", "minutes:seconds" "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
- `-j`, `--json`: Read a .json input file.
- `-v`, `--verbose`: Increases logging level from WARNING to -v for INFO or -vv for DEBUG
- `-h`, `--help`: Show help message and exit.

Detailed explanation is available in [doc/BQP_INPUT_EXAMPLE.md](doc/BQP_INPUT_EXAMPLE.md).

---

### Examples

Please check the following Python files to find how to setup biqbin solver through Python

- [`biqbin_maxcut.py`](biqbin_maxcut.py): Example on how to run the default version of biqbin.
- [`biqbin_qubo.py`](biqbin_qubo.py): Example on how to run QUBO problem.
- [`biqbin_heuristic.py`](biqbin_heuristic.py): Example on how to custom heuristc for lower bound estimation.
- [`biqbin_bqp.py`](biqbin_bqp.py): Example on how to run general BQP biqbin.

---

> **NOTE:** The default **maximum problem size** is **1024 nodes**.  
> If you require more, **edit** the value of `NMAX` in `biqbin_cpp_api.h`.

---

## Problem Instance Input files

> **NOTE:** Biqbin can only solve problem instances with **integer edge weight**!  
> This applies both to the Maxcut and Qubo solvers.

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

---

## Contact information

For any inquiries about the software provided use the following contacts:

- Beno Zupanc: [Github account](https://github.com/Zvmcevap) or [email](beno.zupanc@rudolfovo.eu)
