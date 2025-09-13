# BiqBin: A Solver for Max-Cut and Unconstrained Binary Quadratic Problem

## **Quantum solver for hard BInary Quadratic problems (QBIQ)**

### **Aris Project (L1-60136),  1.1.2025 - 31.12.2027**
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


##  Setup

>**Note:** This readme is for PC example, for HPC please refer to [HPC_EXAMPLE.md](HPC_EXAMPLE.md)

#### Biqbin requires **Linux operating system**. 

It also requires:
- Python development headers (the `python3-dev` package).
- MPI implementation (`OpenMPI` or `MPICH`)
- `OpenBLAS` mathematics library

The rest can be installed with `pip install -r requirements.txt`:
```
pybind11
scipy
numpy
dwave-neal 
```

Below are description on two seperate setup instruction, either [**Anaconda**](#conda-based-build) or [**Docker**](#docker) based builds.

### Conda based build
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
C and C++ compiler
```bash
conda install -c conda-forge gxx
```
MPI
```bash
conda install conda-forge::openmpi
```
OpenBLAS
```bash
conda install -c conda-forge openblas
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

##  Docker

**`Dockerfile`** available for running the solver in a docker container, image can be created using Makefile command

```bash
make docker
```

To access the docker container use:
```bash
make docker-shell 
```
for other specific commands please checkout the `Makefile`

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
mpirun -n N python3 biqbin_maxcut.py problem_instance [-p PARAMS] [-w] [-o OUTPUT]
```

- `N`: number of processes to run the program using MPI, program needs at least 3 (1 master, and 2 worker process) to be used.
- `problem_instance`: A `JSON` serializable sparse coo adjacency matrix.
- `-p PARAMS`: Optional custom parameter file used to configure the solver, defaults to 'params'.
- `-w`, `--overwrite`: Optional command to overwrite the output file if one already exists instead of appending '_NUMBER'.
- `-o OUTPUT`, `--output OUTPUT`: Optional custom OUTPUT file name.
- `-e`, `--edge_weight`: Optional command to use edge weight `problem_instance` that C-only solver uses.
- `-t TIME`, `--time TIME`: Set running time limit; acceptable time formats include "minutes", "minutes:seconds" "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
- `-h`, `--help`: Show help message and exit.
---

### Python Wrapper for QUBO-s

Example:
```bash
mpirun -n 3 python3 biqbin_qubo.py tests/qubos/40/kcluster40_025_10_1.json params
```
General command:
```bash
mpirun -n N python3 biqbin_qubo.py problem_instance [-p PARAMS] [-w] [-o OUTPUT]
```

- `N`: number of processes to run the program using MPI, program needs at least 2 (1 master, and 1 worker process) to be used.
- `problem_instance`: `JSON` serializable dictionary containing "qubo" key and a sparse coo matrix for value (see tests/qubo/) folder for examples.
- `-p PARAMS`: Optional custom parameter file used to configure the solver, defaults to 'params'.
- `-w`, `--overwrite`: Optional command to overwrite the output file if one already exists instead of appending '_NUMBER'.
- `-o OUTPUT`, `--output OUTPUT`: Optional custom OUTPUT file name.
- `-O`, `--optimize`: Divide QUBO values by their GCD.
- `-t TIME`, `--time TIME`: Set running time limit; acceptable time formats include "minutes", "minutes:seconds" "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
- `-h`, `--help`: Show help message and exit.

---
### QUBO DWaveSampler
In `biqbin_heuristic.py` there is an example of a `QuboDwaveSampler` class which inhearits from `QUBOSolver` which overrides
Biqbins default heuristic. It takes an extra `sampler` argument plus any arguments Dwaves sampler takes.
In our example we use DWaves `neal.SimulatedAnnealingSampler`.

Example:
```bash
mpirun -n 3 python3 biqbin_heuristic.py tests/qubos/40/kcluster40_025_10_1.json params
```
General command:
```bash
mpirun -n N python3 biqbin_heuristic.py problem_instance [-p PARAMS] [-w] [-o OUTPUT] [-i] [-d]
```

- `N`: number of processes to run the program using MPI, program needs at least 2 (1 master, and 1 worker process) to be used.
- `problem_instance`: `JSON` serializable dictionary containing "qubo" key and a sparse coo matrix for value (see tests/qubo/) folder for examples.
- `-p PARAMS`: Optional custom parameter file used to configure the solver, defaults to 'params'.
- `-w`, `--overwrite`: Optional command to overwrite the output file if one already exists instead of appending '_NUMBER'.
- `-o OUTPUT`, `--output OUTPUT`: Optional custom OUTPUT file name.
- `-d`, `--debug`: Enables debug logs
- `-i`, `--info`: Enable info logs
- `-t TIME`, `--time TIME`: Set running time limit; acceptable time formats include "minutes", "minutes:seconds" "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
- `-h`, `--help`: Show help message and exit.

### Examples

Please check the following Python files to find how to setup biqbin solver through Python

- `biqbin_maxcut.py`: Example on how to run the default version of biqbin.
- `biqbin_qubo.py`: Example on how to run QUBO problem.
- `biqbin_heuristic.py`: Example on how to custom heuristc for lower bound estimation.


---

> **NOTE:** The default **maximum problem size** is **1024 nodes**.  
> If you require more, **edit** the value of `NMAX` in `biqbin_cpp_api.h`.

---

## Input files

> **NOTE:** Biqbin can only solve problem instances with  **integer edge weight**!  
> This applies both to the Maxcut and Qubo solvers.

Below are example of instance problem files for Maxcut (C and python versions) and Qubos (python version).

### `DataGetter class`
Python versions of the solver use a DataGetter class, split into seperate subclasses for different types of input instances/problems:

- `DataGetterMaxCutDefault` uses the default C implementation of parsing the file, thus requires the same edge weight format.
- `DataGetterAdjacencyJson` parses a `json` serializable dictionary with "adjacency" key and a `scipy.coo_matrix` of an adjacency matrix as value.
- `DataGetterJson` is the default `DataGetter` for QUBO instances, parses a `scipy.coo_matrix` of a QUBO into a MaxCut adjacency matrix.

#### MaxCut default input example
C-version expects an edge weight list format, with the first line containing the number of vertices and edges:

```
3 2
1 2 3
1 3 -4
```
An example of a problem instance with 3 vertices and 2 edges, edge (1, 2) with weight of 3 and edge (1, 3) with weight of -4.

#### MaxCut adjacency json input example
Json serializable dictionary with "adjacency" key and a `scipy.coo_matrix` of an adjacency matrix as value:

```json
{"adjacency": {"shape": [2, 2], "nnz": 4, "row": [0, 0, 1, 1], "col": [0, 1, 0, 1], "data": [-1, 3, 3, -1]}}
```

#### Qubo input example

Qubo solver expects a json file with a "qubo" key that has a `scipy.coo_matrix` as value:

```json
{"qubo": {"shape": [2, 2], "nnz": 4, "row": [0, 0, 1, 1], "col": [0, 1, 0, 1], "data": [-1, 3, 3, -1]}}
```

Other key, value pairs can be added per users discretion.
---

> **NOTE (AGAIN):** Biqbin can only solve problem instances with  **integer edge weight**!  
> This applies all versions of the solver.


#### Utils

A converter `utils.py` which takes in a dense QUBO 2D list or numpy array and returns a json serializable dictionary in the proper form:

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

def qubo_to_biqbin_representation(qubo) -> dict:
    """Converts a dense qubo represantation 2D array to the expected biqbin format of a json serializable 
    dict with 'qubo' key and a sparse qubo represantation as value.

    Args:
        qubo: 2D list or numpy array

    Returns:
        dict: json serializable dictionary that Biqbin can parse. Save to file and pass the path to DataGetterJson.
    """
    if not np.all(np.asarray(qubo) % 1 == 0):
        raise ValueError("All QUBO values need to be integers!")

    return {
        'qubo': to_sparse(qubo)
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