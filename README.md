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

## 🔧 BiqBin Setup Requirements (Ubuntu 22.04)

### 📦 System Dependencies:

- `build-essential`
- `libopenblas-dev`
- `mpich`
- `python3`
- `python3-pip`

### 📦 Python Packages:

- `numpy`
- `pytest`: for unit tests

## ⚙️ Installation

1. Open the `Makefile`.
2. Set the **compiler** and **BLAS/LAPACK** package paths according to your system.
3. Run:

```bash
make
```

### 🔧 Docker

Dockerfile available for running the solver in a docker container, image can be created using Makefile command

```bash
make docker
```

---

## 🚀 Usage

### Original Biqbin Maxcut Parallel solver - C only

Can be run with

```bash
mpiexec -n num_processess ./biqbin instance_file params
```

- `num_processes`: number of processes to run the program using MPI, program needs at least 2 (1 master, and 1 worker process) to be used.
- `instance_file`: A file containing the graph (in edge list format).
- `params`: The parameter file used to configure the solver.

To run the g05_60.0 instance use this make command

```bash
make run
```

Or the following for to run all g05\_\* instances

```bash
make run-all
```

---

### Python Wrapper for Biqbin Maxcut Parallel solver

Can be run with

```bash
mpiexec -n num_processes python3 run_example.py instance_file params
```

- `num_processes`: number of processes to run the program using MPI, program needs at least 2 (1 master, and 1 worker process) to be used.
- `instance_file`: A file containing the graph (in edge list format).
- `params`: The parameter file used to configure the solver.

To run the g05_60.0 instance use this make command

```bash
make run-python
```

Or the following for to run all g05_X instances

```bash
make run-python-all
```

---

> ⚠️ **NOTE:** The default **maximum problem size** is **1024 nodes**.  
> If you need more, **edit** the value of `NMAX` in `biqbin.h` and `biqbin_data_objects.py`.

---

## Explanation of the Python Wrapper

For easiest overview into how the wrapper is setup and run in code, please see `run_example.py` file.

### **Class: `ParallelBiqbin`**

The `ParallelBiqbin` Python class serves as a Python wrapper for interacting with the C-based `biqbin` library.

---

#### **Mutual Methods**

- **`__init__()`**  
  Initializes the class by loading the shared C library (`biqbin.so`). Sets up function pointers to C functions.

- **`biqbin.init_MPI(graph_path, params_path)`**  
  Initializes MPI in the C-solver. Returns `rank`. **⚠️Must be run first!⚠️**

---

#### **Master Process Methods**

- **`master_init(filename, L, num_verts, num_edge, parameters) -> bool`**  
  Initializes the master process (rank 0), sets parameters, the initial problem (SP), and communicates it to other workers in the C-solver.  
  Input arguments:

  - `filename`: Byte string (e.g., `b"filename"` or `filename.encode("utf-8")`) used to open the output file. The content is not read.
  - `L`: Laplacian matrix in a `numpy.ndarray` with `numpy.float64` dtype.
  - `num_verts`: int — number of vertices in the graph.
  - `num_edge`: int — number of edges in the graph.
  - `parameters`: Instance of `BiqBinParameters`.

  **Returns:** `bool` — `over`, indicating if the solver is done.

- **`master_main_loop() -> bool`**  
  Continuously checks for solver progress **while `over` is False**, and coordinates communication and computation with worker nodes.

- **`master_end()`**  
  Notifies workers that solving is `over` Performs cleanup and then terminates the process.

#### **Worker Process Methods**

- **`worker_init(parameters) -> bool`**  
  Initializes the worker process, sets parameters, receives the initial problem (SP) from **master process** in the C-solver.  
  Input arguments:

  - `parameters`: Instance of `BiqBinParameters`.

  **Returns:** `bool` — `over`, indicating if the solver is done.

- **`worker_main_loop(rank: int) -> bool`**  
  Runs **while `over` is False**, execution happens in multiple steps:

  - **`over = self.biqbin.worker_check_over()`**: Waits for **over** signal, returns `True` if it is.
  - **`worker_receive_problem()`**: Receives problem from another worker or master process, places it into its priority queue.
  - **`while self.biqbin.isPQEmpty() == 0`**: Loops while the priority queue in C-solver is not empty
    - **`if self.biqbin.time_limit_reached()`**: Checks if time limit is reached, if it is returns `True`.
    - **`babnode = self.biqbin.Bab_PQPop()`**: Pops a **BabNode** that needs evaluating from the C-solvers priority queue.
    - **`old_lb = self.biqbin.Bab_LBGet()`**: Saves current best lower bound, because the evaluation updates it.
    - **`self.biqbin.evaluate_node_wrapped(babnode, rank)`**: Evaluate the popped node, updates best solution.
    - **`self.biqbin.after_evaluation(babnode, old_lb)`**: Compares previous best solution with the new one, communicates with master if it needs updating, frees node from memory.
  - **`self.biqbin.worker_send_idle()`**: Send IDLE signal master process, letting it know the work is finished for this process
  - **`return False`** to continue another run of `worker_main_loop`.

- **`worker_end()`**  
  Performs cleanup and then terminates the process.

---

### **Class: HelperFunctions**

The `HelperFunctions` class helps parse the graph input and params files located in the repository (either in `test/Instances` or `Instances` folders) to be passed into `ParallelBiqbin` class methods.

#### **Key Methods:**

- **`read_maxcut_input(file_path: str)`**  
  Reads the graph file at `file_path` location and returns a touple of:

  - `Adj`: adjacency matrix of the graph as a numpy array.
  - `num_vertices`: number of vertices in the graph.
  - `num_edges`: number of edges in the graph.
  - `name`: filename encoded into a byte string

- **`get_SP_L_matrix(Adj)`**  
  Receives an adjacency matrix, constructs and returns the L matrix that the C-solver expects in `master_init` method in `ParallelBiqbin` class.

- **`read_parameters_file(file_path: str)`**  
  Reads the parameters file and returns a `BiqBinParameters` object that can be passed into `master_init` and `worker_init`.

---

## 🛠️ Explanation of Parameters

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
| `detailed_output`   | If `1`, enables **verbose output** for each B&B node                           |

---
