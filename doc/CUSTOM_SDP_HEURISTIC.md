# Custom SDP Bounds and Heuristics

BiqBin's Python solver classes allow branch-and-bound node evaluation to be customized from Python.

The main extension points are:

| Method                   | Used for                                                           |
| ------------------------ | ------------------------------------------------------------------ |
| `sdp_bound(...)`         | SDP / upper-bound computation on regular branch-and-bound nodes    |
| `initial_sdp_bound(...)` | SDP / upper-bound computation on the root node                     |
| `heuristic(...)`         | Heuristic / lower-bound solution on regular branch-and-bound nodes |
| `initial_heuristic(...)` | Heuristic / lower-bound solution on the root node                  |

`QUBOSolver` and `BQPSolver` inherit the same callback mechanism from `MaxCutSolver`. Their callbacks operate on the transformed Max-Cut problem used internally by BiqBin.

> **Important:** BiqBin is an exact solver. A custom SDP callback must return a valid upper bound for the current subproblem. Returning a bound that is too low can prune a branch containing the optimum and invalidate the final result.

---

## Callback selection

The root node and the remaining branch-and-bound nodes can use different implementations.

On MPI rank 0, BiqBin uses:

```python
self._sdp_bound_fn = self.initial_sdp_bound
self._heuristic_fn = self.initial_heuristic
```

On worker ranks, BiqBin uses:

```python
self._sdp_bound_fn = self.sdp_bound
self._heuristic_fn = self.heuristic
```

The default root methods delegate to the regular methods:

```python
def initial_sdp_bound(self, node, P0, P, *args, **kwargs):
    return self.sdp_bound(node, P0, P, *args, **kwargs)

def initial_heuristic(self, L, *args, **kwargs):
    return self.heuristic(L, *args, **kwargs)
```

Therefore:

- override only `sdp_bound()` to replace the SDP bound everywhere;
- override `initial_sdp_bound()` as well if the root should use a different SDP strategy;
- override only `heuristic()` to replace the heuristic everywhere;
- override `initial_heuristic()` as well if the root should use a different heuristic.

If an `initial_estimate` is supplied, it is used instead of `initial_heuristic()` on the root node.

---

## Custom SDP bound

The callback signature is:

```python
def sdp_bound(
    self,
    node: BabNode,
    P0: Problem,
    P: Problem,
    *args,
    **kwargs,
) -> float:
    ...
```

where:

- `node` is the current branch-and-bound node;
- `P0` is the original full Max-Cut problem;
- `P` is the current reduced subproblem.

A custom implementation must:

1. compute a valid upper bound for `P`;
2. compute and register the corresponding SDP primal solution matrix;
3. return a finite real scalar (float).

Example:

```python
class CustomSDPSolver(MaxCutSolver):
    def sdp_bound(self, node, P0, P, *args, **kwargs):
        bound, X = my_sdp_solver(P.L)

        X = np.asarray(X, dtype=np.float64, order='C')
        self.set_sdp_primal_solution(X) # set the SDP primal solution matrix with Biqbin

        return float(bound)
```

`P.L` is backed by native solver memory. If an external library may modify its input, pass a copy.

### Setting the SDP primal solution

When replacing the default SDP routine, calling:

```python
self.set_sdp_primal_solution(X)
```

is mandatory.

The primal matrix is used for branching and is also required by the default Goemans-Williamson heuristic.

`X` must:

- have shape `(P.n, P.n)`;
- contain finite values;
- have values in `[-1, 1]`;
- be C-contiguous when passed to the native layer.

A safe conversion is:

```python
X = np.asarray(X, dtype=np.float64, order='C')
self.set_sdp_primal_solution(X)
```

The built-in `sdp_bound()` handles the native primal solution automatically.

---

## Root-only SDP bound

To use a special SDP implementation only on the root node, override `initial_sdp_bound()` and leave `sdp_bound()` unchanged:

```python
class RootSDPSolver(MaxCutSolver):
    def initial_sdp_bound(self, node, P0, P, *args, **kwargs):
        bound, X = expensive_root_sdp(P.L)

        X = np.asarray(X, dtype=np.float64, order='C')
        self.set_sdp_primal_solution(X)

        return float(bound)
```

The root node uses `expensive_root_sdp()`. Worker nodes continue using BiqBin's default SDP bound.

A common strategy is to spend more time obtaining a stronger root bound and use a cheaper bound deeper in the tree.

You can also use two custom strategies:

```python
class TwoLevelSDPSolver(MaxCutSolver):
    def initial_sdp_bound(self, node, P0, P, *args, **kwargs):
        bound, X = strong_root_sdp(P.L)
        self.set_sdp_primal_solution(
            np.asarray(X, dtype=np.float64, order='C')
        )
        return float(bound)

    def sdp_bound(self, node, P0, P, *args, **kwargs):
        bound, X = faster_node_sdp(P.L)
        self.set_sdp_primal_solution(
            np.asarray(X, dtype=np.float64, order='C')
        )
        return float(bound)
```

---

## Custom heuristic

The heuristic callback signature is:

```python
def heuristic(
    self,
    L: np.ndarray,
    *args,
    **kwargs,
):
    ...
```

The following objects are supplied through `kwargs`:

```python
node = kwargs['node']
P0 = kwargs['P0']
P = kwargs['P']
```

The callback must return a one-dimensional binary vector containing one value for each free decision variable of the current subproblem. The wrapper currently validates the vector against shape:

```python
(P.n - 1,)
```

and requires all values to be `0` or `1`.

Example:

```python
class CustomHeuristicSolver(MaxCutSolver):
    def heuristic(self, L, *args, **kwargs):
        x = my_heuristic(L)
        return np.asarray(x, dtype=np.int32)
```

A custom heuristic does not return the objective value. BiqBin evaluates the returned binary vector, updates the lower-bound solution, and records the objective internally.

### Relationship to the SDP callback

The default native SDP routine can invoke the heuristic internally.

A custom SDP routine normally does not. After the custom SDP callback returns, the Python node-evaluation wrapper detects that the heuristic was not called and invokes the configured heuristic automatically.

A normal custom SDP implementation therefore does not need to call `self.heuristic(...)` manually.

---

## Root-only heuristic

To use a different heuristic on the root node:

```python
class RootHeuristicSolver(MaxCutSolver):
    def initial_heuristic(self, L, *args, **kwargs):
        x = expensive_root_heuristic(L)
        return np.asarray(x, dtype=np.int32)
```

If `heuristic()` is not overridden, worker nodes continue using the default Goemans-Williamson heuristic.

To use one custom heuristic on the root and another on worker nodes:

```python
class TwoLevelHeuristicSolver(MaxCutSolver):
    def initial_heuristic(self, L, *args, **kwargs):
        return np.asarray(
            expensive_root_heuristic(L),
            dtype=np.int32,
        )

    def heuristic(self, L, *args, **kwargs):
        return np.asarray(
            fast_node_heuristic(L),
            dtype=np.int32,
        )
```

> **Initial estimates:** if the solver is constructed with `initial_estimate=...`, the estimate replaces the normal `initial_heuristic()` path on the root. `heuristic()` is still used for later branch-and-bound nodes.

---

## Using custom SDP and heuristic callbacks together

All four callbacks can be overridden independently:

```python
class CustomSolver(MaxCutSolver):
    def initial_sdp_bound(self, node, P0, P, *args, **kwargs):
        bound, X = root_sdp(P.L)
        self.set_sdp_primal_solution(
            np.asarray(X, dtype=np.float64, order='C')
        )
        return float(bound)

    def sdp_bound(self, node, P0, P, *args, **kwargs):
        bound, X = node_sdp(P.L)
        self.set_sdp_primal_solution(
            np.asarray(X, dtype=np.float64, order='C')
        )
        return float(bound)

    def initial_heuristic(self, L, *args, **kwargs):
        return np.asarray(
            root_heuristic(L),
            dtype=np.int32,
        )

    def heuristic(self, L, *args, **kwargs):
        return np.asarray(
            node_heuristic(L),
            dtype=np.int32,
        )
```

If the same implementation should be used on every node, only override the regular methods:

```python
class CustomSolver(MaxCutSolver):
    def sdp_bound(self, node, P0, P, *args, **kwargs):
        bound, X = my_sdp(P.L)
        self.set_sdp_primal_solution(
            np.asarray(X, dtype=np.float64, order='C')
        )
        return float(bound)

    def heuristic(self, L, *args, **kwargs):
        return np.asarray(
            my_heuristic(L),
            dtype=np.int32,
        )
```

The default `initial_sdp_bound()` and `initial_heuristic()` delegate to these overridden methods.

---

## Callback data available from Python

The native `BabNode` and `Problem` structures are exposed to Python callbacks.

### `BabNode`

Useful fields include:

```python
node.xfixed
node.sol.x
node.fracsol
node.level
node.upper_bound
```

- `node.xfixed` indicates which original variables are fixed at the current branch-and-bound node.
- `node.sol.x` contains the current binary solution values.
- `node.fracsol` contains the fractional solution derived from the SDP primal solution and is used for branching.
- `node.level` is the branch-and-bound depth.
- `node.upper_bound` is the node's current upper bound.

Treat arrays exposed from native structures as read-only solver state. Copy them before passing them to code that may mutate inputs.

### `Problem`

Useful fields include:

```python
P.L
P.n
P.NIneq
P.NPentIneq
P.NHeptaIneq
P.bundle
```

`P0` represents the original problem while `P` represents the current subproblem.

The most commonly useful values are:

```python
P0.L   # original objective matrix
P0.n   # original problem size

P.L    # current subproblem objective matrix
P.n    # current subproblem size
```

---

## QUBO and BQP callbacks

`QUBOSolver` and `BQPSolver` inherit their callback behavior from `MaxCutSolver`.

Therefore `sdp_bound()`, `initial_sdp_bound()`, `heuristic()`, and `initial_heuristic()` operate on BiqBin's internal **Max-Cut representation**, not directly on the original QUBO or BQP formulation.

Example:

```python
class CustomQUBOSolver(QUBOSolver):
    def sdp_bound(self, node, P0, P, *args, **kwargs):
        # P.L is the transformed Max-Cut subproblem.
        bound, X = my_sdp(P.L)
        self.set_sdp_primal_solution(
            np.asarray(X, dtype=np.float64, order='C')
        )
        return float(bound)
```

The same applies to `BQPSolver`.

Initial estimates are handled separately by the solver classes and converted into the internal Max-Cut representation before root-node evaluation.

---

## Root callback data collection

Timing and return-value data can be collected for root callbacks:

```python
solver = CustomSolver(
    problem=problem,
    params='params',
    collect_heuristic_root_data=True,
    collect_sdp_bound_root_data=True,
)
```

Each collected callback entry contains:

```python
{
    'time': ...,
    'value': ...,
}
```

The resulting metadata includes:

```python
solution.meta_data['root_node']['heuristic_data']
solution.meta_data['root_node']['sdp_bound_data']
```

Heuristic root metadata also contains accumulated heuristic time and call count.

---

## Error handling under MPI

Python callbacks execute while the native MPI solver is running.

Invalid callback results or exceptions are fatal to the distributed solve. BiqBin aborts the MPI job rather than allowing one rank to fail while the others remain blocked in MPI communication.

Typical fatal callback errors include:

- returning a non-finite or non-scalar SDP bound;
- failing to call `set_sdp_primal_solution()` from a custom SDP callback;
- supplying a primal matrix with the wrong shape or invalid values;
- returning a heuristic vector with the wrong length;
- returning non-binary heuristic values;
- raising an exception inside a callback.

Validate external-library inputs before expensive work and avoid mutating native-backed callback arrays.

---

## Minimal complete example

```python
import numpy as np

from biqbin import QUBOSolver, init, ProblemQubo


class MySolver(QUBOSolver):
    def sdp_bound(self, node, P0, P, *args, **kwargs):
        bound, X = my_sdp_solver(P.L)

        X = np.asarray(X, dtype=np.float64, order='C')
        self.set_sdp_primal_solution(X)

        return float(bound)

    def heuristic(self, L, *args, **kwargs):
        x = my_heuristic(L)
        return np.asarray(x, dtype=np.int32)


init() # initialize MPI
problem = ProblemQubo(
    Q=np.array([[1, -2],[0, 4]]),
    offset=0,
    problem_name='example_problem',
    is_minimization=True
)
solver = MySolver(
    problem=problem,
    params='params',
)

solution = solver.compute()

if solution is not None:
    print(solution)
```

Run the program through MPI in the same way as the standard BiqBin Python entry points.

---

## Recommended override patterns

### Replace SDP everywhere

Override:

```python
sdp_bound()
```

The default `initial_sdp_bound()` delegates to it.

### Replace only root SDP

Override:

```python
initial_sdp_bound()
```

Leave `sdp_bound()` unchanged.

### Replace heuristic everywhere

Override:

```python
heuristic()
```

The default `initial_heuristic()` delegates to it.

### Replace only root heuristic

Override:

```python
initial_heuristic()
```

Leave `heuristic()` unchanged.

### Use custom root and non-root implementations

Override the required pairs:

```python
initial_sdp_bound()
sdp_bound()

initial_heuristic()
heuristic()
```

---

## Notes for exactness

Custom heuristics can be weak without invalidating exactness; they mainly affect lower-bound quality and solver performance.

Custom SDP bounds are different: they participate directly in branch pruning. A custom bound must be a mathematically valid upper bound for the current Max-Cut subproblem.

When developing a custom bound, compare it against known instances and BiqBin's default SDP implementation before relying on it for exact solves.
