# Biqbin Custom Solver Guide

## Overview

`biqbin` exposes four callback methods you can override in a subclass of `MaxCutSolver` or `QUBOSolver` to plug in custom SDP bounds and heuristics at different points in the branch-and-bound tree.

| Method | When it runs | Purpose |
|---|---|---|
| `initial_sdp_bound` | Root node only | Lower bound (SDP relaxation) |
| `initial_heuristic` | Root node only | Upper bound (feasible solution) |
| `sdp_bound` | All non-root nodes | Lower bound (SDP relaxation) |
| `heuristic` | All non-root nodes | Upper bound (feasible solution) |

If `initial_sdp_bound` is not overridden, `sdp_bound` is used on the root node too. Same applies to `initial_heuristic` / `heuristic`.

---

## The `set_sdp_primal_solution` Requirement

Biqbin requires a primal SDP solution matrix (a PSD matrix `X` in {-1, 1} range) to be set before it runs. The default `sdp_bound` sets this internally. 

**If you override `sdp_bound`, you must call `self.set_sdp_primal_solution` before returning.**

```python
def sdp_bound(self, node, P0, P, *args, **kwargs) -> float:
    # ... your SDP routine ...
    X = np.identity(P.n) # your PSD matrix of shape (P.n, P.n)
    self.set_sdp_primal_solution(X)
    return my_sdp_value
```

---

## Method Signatures

```python
def initial_sdp_bound(self, node: BabNode, P0: Problem, P: Problem, *args, **kwargs) -> float:
    ...

def sdp_bound(self, node: BabNode, P0: Problem, P: Problem, *args, **kwargs) -> float:
    ...

def initial_heuristic(self, L: np.ndarray, *args, **kwargs) -> npt.NDArray[np.int32]:
    ...

def heuristic(self, L: np.ndarray, *args, **kwargs) -> npt.NDArray[np.int32]:
    ...
```

- `node` — current B&B node
- `P0` — original (full) problem (`P0.n` is the size of the full problem `P0.L` is the Laplacean of the full problem)
- `P` — subproblem at the current node (`P.n` is its size, `P.L` is its Laplacian)
- `L` — Laplacian matrix passed to heuristics; returned vector must be of length `L.shape[0] - 1` (`P.n - 1`)

Heuristic return values must be integer binary vectors (`dtype=np.int32`).

---

## Examples

```python
class CustomSolver(QUBOSolver):
    def initial_sdp_bound(self, node, P0, P, *args, **kwargs) -> float:
        # Compute the SDP relaxation value and SDP primal solution X
        self.set_sdp_primal_solution(X) # set it before leaving the function
        return sdp_value
        
    def initial_heuristic(self, L, *args, **kwargs) -> npt.NDArray[np.int32]:
        # Run heuristic on root and return a binary 0-1 solution vector of size L.shape[0] - 1
        x = np.zeros(L.shape[0] - 1, dtype=np.int32)
        return x

    def sdp_bound(self, node, P0, P, *args, **kwargs) -> float:
        return sdp_value # replaced by heuristic, so no primal needed

    def heuristic(self, L, *args, **kwargs) -> npt.NDArray[np.int32]:
        # Run heuristic and return a binary 0-1 solution vector of size L.shape[0] - 1
        x = np.zeros(L.shape[0] - 1, dtype=np.int32)
        return x
```
