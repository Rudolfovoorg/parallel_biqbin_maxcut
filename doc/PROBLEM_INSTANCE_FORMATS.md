# Biqbin Problem Instance Formats

Biqbin support different file formats for both **Max-Cut** and **Qubo** solvers:

- `JSON` dictionary with `scipy.sparse.coo_matrix`
- `MatrixMarket` format, both [sparse and dense](https://math.nist.gov/MatrixMarket/formats.html)
- `Edge-Weight list` found also in the [Stanford GSet](https://web.stanford.edu/~yyye/yyye/Gset/)

QUBO only:

- `QPLIB` file format for "QBN" instances, found [here](https://qplib.zib.de/).

> **Note:** Biqbin only supports **integer** values for weights or matrix values in all versions!

### CLI command examples

By default both solvers try to load the [JSON format](#json-input-example) format, but this can be set with the `--format` optional parameter:

```bash
mpirun -n 3 python3 biqbin_qubo.py matrixmarket_problem_path --format mm
```

`--format` valid options are:
- `json` for the default [JSON format](#json-input-example)
- `mm` for the [MatrixMarket format](#matrix-market-format)
- `ew` for the [Edge-Weight list format](#edge-weight-list-input-example)

QUBO also supports:
- `qplib` for the [qplib file format](#qplib-format-for-qubos)


## Python Example

There is a seperate python class for each of the supported formats split between Max-Cut and QUBO data parsers. Implementation is located in [biqbin/data_parsers.py](../biqbin/data_parsers.py)

```py
# All importable data parsers and solvers
from biqbin import (MaxCutFromJson, # For MaxCut problems
                    MaxCutFromMatrixMarket,
                    MaxCutFromEdgeWeights,
                    MaxCutSolver,
                    QuboFromJson, # For QUBO problems
                    QuboFromMatrixMarket,
                    QuboFromEdgeWeights,
                    QuboFromQPLIB,
                    QUBOSolver)

# They are all used in the same way:
data_parser = QuboFromJson(filename) # instantiate with the path to the problem file
problem = data_parser.read()         # call the read() method, which returns a 'Problem' instance for the solver

solver = QUBOSolver(problem_qubo)    # Instantiate the solver with the returned 'Problem' instance
solution = solver_qubo.compute()     # Compute the solution
```


## Edge-Weight List input example

> **NOTE:** This is the only file format that the C-Version (biqbin_executable) accepts.

Edge weight list format, with the first line containing the number of vertices and edges followed by a space-seperated list of two nodes and weight:

```
3 2
1 2 3
1 3 -4
```

An example of a problem instance with 3 vertices and 2 edges, edge (1, 2) with weight of 3 and edge (1, 3) with weight of -4.

## Json input example

Json serializable dictionary with either a "maxcut" or "qubo" key and a `scipy.sparse.coo_matrix` as value:

MaxCut:

```json
{
  "maxcut": {
    "shape": [2, 2],
    "nnz": 4,
    "row": [0, 0, 1, 1],
    "col": [0, 1, 0, 1],
    "data": [-1, 3, 3, -1]
  }
}
```

QUBO example, it allows optional values:
- an offset (a constant value added to the objective value)
- and the objective sense (true for minimization, false for maximization)

```json
{
  "qubo": {
    "shape": [2, 2],
    "nnz": 4,
    "row": [0, 0, 1, 1],
    "col": [0, 1, 0, 1],
    "data": [-1, 3, 3, -1]
  },
  "offset": 0.0,
  "is_minimization": true
}
```

## Matrix Market format

We support both the coordinate and array format, as [documented here](https://math.nist.gov/MatrixMarket/formats.html).

```
%%MatrixMarket matrix coordinate integer symmetric
5 5 8
1 1 1
2 2 1
3 3 1
1 4 6
4 2 2
4 4 -2
4 5 3
5 5 1
```

## QPLIB format for QUBOs

Qubo problem instances can be loaded from the qplib file, [documented here](https://qplib.zib.de/).
Please note that we only support *"QBN"* or *"LBN"* problems, which stands for "Quadratic", "Binary", "No-constraints".


```
tests/qubos/40/kcluster40_025_10_1.json
QBN
minimize
40 # number of variables
780 # number of quadratic terms in the objective
2 1 440.0
3 1 440.0
3 2 440.0
...
0.0 # default value of linear coefficients in objective
40 # number of non-default linear coefficients in objective
1 -595.0
2 -595.0
3 -595.0
...
0.0 # objective constant
1.79769313486232E+308 # value for infinity
0.0 # default variable primal value in starting point
0 # number of non-default variable primal values in starting point
0.0 # default variable bound dual value in starting point
0 # number of non-default variable bound dual values in starting point
0 # number of non-default variable names
0 # number of non-default constraint names
```

### `ToFile` class

Python versions of the solver use a `ToFile` class, split into seperate subclasses to read different types of input formats, instances/problems.
`ToFile` derived class needs to implement `read(filename: str)` method that returns and instance of a `ProblemMaxCut` class (or it's subclass, such as `ProblemQubo` class).


## BQP input example

BQP input is explained [here](BQP_INPUT_EXAMPLE.md).