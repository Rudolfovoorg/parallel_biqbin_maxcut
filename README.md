# BiqBin: A Solver for Max-Cut and Unconstrained Binary QP

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

##  BiqBin Setup Requirements (Ubuntu 22.04)

###  System Dependencies:

- `build-essential`
- `libopenblas-dev`
- `mpich`
- `python3`
- `python3-pip`

###  Python Packages:

- `numpy`
- `pytest`: for unit tests

##  Installation

1. Open the `Makefile`.
2. Set the **compiler** and **BLAS/LAPACK** package paths according to your system.
3. Run:

```bash
make
```

### Docker

Dockerfile available for running the solver in a docker container, image can be created using Makefile command

```bash
make docker
```

To access the docker container

```bash
make docker-run
```

To clean up the docker container and image

```bash
make docker-clean
```
---

## Usage

Can be run with

```bash
mpiexec [-n num_processes] python3 run_example.py instance_file params
```

- `-n num_processes`(Optional): Number of processes to run the program using MPI, program needs at least 2 (1 master, and 1 worker process) to be used.
- `instance_file`: A file containing the graph (in edge list format).
- `params`: The parameter file used to configure the solver.

To run the g05_60.0 instance use this make command

```bash
make run
```

For more commands see `Makefile`.

---

> ⚠️ **NOTE:** The default **maximum problem size** is **1024 nodes**.  
> If you need more, **edit** the value of `NMAX` in `biqbin.h` and `biqbin_data_objects.py`.

---

## Explanation of the Python Wrapper

- Example on how the wrapper is setup and run in python, please see **`run_example.py`** file.
- Example on how to change the heuristic function please see **`new_heuristics_example.py`**.

### **Class: `ParallelBiqbin`**

The `ParallelBiqbin` Python class serves as the main Python wrapper for interacting with the C-based `biqbin` library. 

It inherits from `BiqbinBase` class which contains all the calls to C functions.
All Python functions that call C functions start with a single underscore (i.e. `_pq_pop`)

---

#### **Key Methods**
- **`__init__(params: BiqBinParameters = BiqBinParameters())`**
  Constructor optionally takes a `BiqBinParameters` class instance or creates a default one for use

- **`compute(graph_path:str)`**
  Runs the solver for the graph instance in the filepath location.

- **`heuristic( babnode: _BabNode, solution_out: np.ndarray, globals: _GlobalVariables)`**  
  By default runs the original `GW_heuristic` function in  `heuristic.c`.
  Can be overriden, but it must use the `@heuristicmethod` decorator.

  `solution_out` is of type `np.ndarray` is a [0, 1] 1D array containing the solution, it will get evaluated after the heuristic is run and used by the solver.
  
- **`read_maxcut_input(self, graph_path: str) -> tuple[np.ndarray, int, int, str]`**
  Reads graph at graph_path and returns a tuple with the following 4 elements:
    - `np.ndarray`: The adjacency matrix.
    - `int`: Number of vertices.
    - `int`: Number of edges.
    - `str`: output file path.

  Function can be overriden to read different kinds of data as long as it returns the adjacency matrix as
  `np.ndarray`, `int` number of vertices, `int` number of edges (this one is only for writing the number of edges in the output)
  and a `str` for the output file path.


- **`get_Laplacian_matrix(Adj: np.ndarray)`**  
  Receives an adjacency matrix as np.ndarray of np.float64, 
  constructs and returns the L matrix that the C-solver expects for the original problem `Problem *SP->L`

---

### **Class: BiqBinParameters**

Class to set or read parameters that can be changed in the original biqbin C implementation.

#### **Key Methods:**

- **`__init__(..., params_filepath=None)`**  
  Reads the graph file at `params_filepath` location if passed in and sets the values to that. 
  For all values that can also be changed during instantiation of the class or later please see the
  **Explanation of Parameters** below.

- **`get_c_struct()`**  
  creates a `ctypes.Structure` needed by `set_parameters` function in C

- **`read_parameters_file(file_path: str)`**  
  Reads the parameters file and sets them in this object instance.

---

## Explanation of Parameters

_(Default values can be found in `biqbin.h`)_

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
| `detailed_output`   | Atm only decides if `BiqBinParamters` get printed to output
---
