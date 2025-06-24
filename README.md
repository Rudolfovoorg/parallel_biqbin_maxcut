# BiqBin: A Solver for Max-Cut and Unconstrained Binary Quadratic Problem

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


##  Setup (Conda-based Build)

This project is fully buildable inside a Conda environment.

### Setup Instructions (Conda Environment)
#### 1. Install Anaconda (if not already)

Download
```bash
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
```
Install
```bash
bash ~/Anaconda3-2024.10-1-Linux-x86_64.sh
```
---
Configure solver if not set to libmamba
```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```
Update to nevest
```bash
conda update -n base -c defaults conda
```

#### 2. Create and activate Conda environment
Create
```bash
conda create -n biqbin-py312 python=3.12
```
Activate
```bash
conda activate biqbin-py312
```

#### 3. Install dependencies
Python.Boost
```bash
conda install conda-forge::libboost-python-devel
```
C and C++ compiler
```bash
conda install -c conda-forge gxx
```
MPI
```bash
conda install conda-forge::openmpi
```
Python packages
```bash
pip install -r requirements.txt
```

#### 5. Compile using Makefile
```bash
make
make test
```

###  Docker

**`Dockerfile`** available for running the solver in a docker container, image can be created using Makefile command

```bash
make docker
```


## Usage

> **NOTE:** Biqbin can only solve problem instances with  **integer edge weight**!  
> This applies both to the Maxcut and Qubo solvers.


> **Min Processes:** Biqbin requires needs at least **3 mpi processes to run**!

> **Over Concurrency:** Depending on your system you must set `OpenBlas` environment variables, to prevent over threading which can **significantly** slow down your system:  
```bash
 export OPENBLAS_NUM_THREADS=1
 export GOTO_NUM_THREADS=1
 export OMP_NUM_THREADS=1
 ```
Setting them 1 is just an example.

### Original Biqbin Maxcut Parallel solver - C only

Run with

```bash
mpirun -n num_processess ./biqbin instance_file params
```

- `num_processes`: number of processes to run the program using MPI, program needs at least 3 (1 master, and 2 worker process) to be used.
- `instance_file`: A file containing the graph (in edge list format).
- `params`: The parameter file used to configure the solver.

---

### Python Wrapper for Biqbin Maxcut Parallel solver

Can be run with

```bash
mpirun -n num_processes python3 biqbin_maxcut.py instance_file params
```

- `num_processes`: number of processes to run the program using MPI, program needs at least 3 (1 master, and 2 worker process) to be used.
- `instance_file`: A file containing the graph (in edge list format).
- `params`: The parameter file used to configure the solver.


---

### Python Wrapper for QUBO-s.

Can be run with

```bash
mpirun -n num_processes python3 biqbin_qubo.py instance_file params
```

- `num_processes`: number of processes to run the program using MPI, program needs at least 2 (1 master, and 1 worker process) to be used.
- `instance_file.json`: dictionary containing "qubo" key and a sparse matrix for value (see tests/qubo/) folder for examples.
- `params`: The parameter file used to configure the solver.

---

### Examples

Please check the following Python files to find how to setup biqbin solver through Python

- `biqbin_maxcut.py`: Example on how to run the default version of biqbin.
- `biqbin_qubo.py`: Example on how to run QUBO problem.
- `biqbin_heuristic.py`: Example on how to custom heuristc for lower bound estimation.


---

> **NOTE:** The default **maximum problem size** is **1024 nodes**.  
> If you require more, **edit** the value of `NMAX` in `biqbin_cpp_api.h`.

---

## Evil QUBOs
A collection of challenging QUBOs is located in `evil_qubos/` folder

---
## Input files

> **NOTE:** Biqbin can only solve problem instances with  **integer edge weight**!  
> This applies both to the Maxcut and Qubo solvers.

Below are example of instance problem files for Maxcut (C and python versions) and Qubos (python version).

### Maxcut
Expects an edge weight list format, with the first line containing the number of vertices and edges:

```
3 2
1 2 3
1 3 -4
```
An example of a problem instance with 3 vertices and 2 edges, edge (1, 2) with weight of 3 and edge (1, 3) with weight of -4.

### Qubo
Qubo solver expects a json file with a "qubo" key that has a `scipy.coo_matrix` as value:

```json
{"qubo": {"shape": [2, 2], "nnz": 4, "row": [0, 0, 1, 1], "col": [0, 1, 0, 1], "data": [-1, 3, 3, -1]}}
```
Other key, value pairs can be added per users discretion.

Python function that takes in a `numpy.array` and turns it into the proper format:

```py
def to_sparse(qubo):
    qubo_sparse_coo = sp.sparse.coo_matrix(qubo)
    return {
        'shape': qubo_sparse_coo.shape, 
        'nnz': qubo_sparse_coo.nnz, 
        'row': qubo_sparse_coo.row.tolist(), 
        'col': qubo_sparse_coo.col.tolist(), 
        'data': qubo_sparse_coo.data.tolist()
    }
```

Another example is available in `qubo_setup_example.py`.

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