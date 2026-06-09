# Biqbin Custom Solver Guide

## Overview

`biqbin` exposes four callback methods you can override in a subclass of `QUBOSolver` to plug in custom SDP bounds and heuristics at different points in the branch-and-bound tree.

| Method | When it runs | Purpose |
|---|---|---|
| `root_sdp_bound` | Root node only | Lower bound (SDP relaxation) |
| `root_heuristic` | Root node only | Upper bound (feasible solution) |
| `sdp_bound` | All non-root nodes | Lower bound (SDP relaxation) |
| `heuristic` | All non-root nodes | Upper bound (feasible solution) |

If `root_sdp_bound` is not overridden, `sdp_bound` is used on the root node too. Same applies to `root_heuristic` / `heuristic`.

---

## The `set_sdp_primal_solution` Requirement

The default `heuristic` needs a primal SDP solution matrix (a PSD matrix `X`) to be set before it runs. The default `sdp_bound` sets this internally. **If you override `sdp_bound` but keep the default `heuristic`, you must call `self.set_sdp_primal_solution(X)` before returning.**

```python
def sdp_bound(self, node, P0, P, *args, **kwargs) -> float:
    # ... your SDP routine ...
    X = np.identity(P.n)          # your PSD matrix of shape (P.n, P.n)
    self.set_sdp_primal_solution(X)
    return my_sdp_value
```

If you also override `heuristic`, you don't need to call `set_sdp_primal_solution` at all.

---

## Method Signatures

```python
def root_sdp_bound(self, node: BabNode, P0: Problem, P: Problem, *args, **kwargs) -> float:
    ...

def sdp_bound(self, node: BabNode, P0: Problem, P: Problem, *args, **kwargs) -> float:
    ...

def root_heuristic(self, L: np.ndarray, *args, **kwargs) -> npt.NDArray[np.int32]:
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

### Override everything (no `set_sdp_primal_solution` needed)

```python
class CustomSolver(QUBOSolver):
    def root_sdp_bound(self, node, P0, P, *args, **kwargs) -> float:
        # Compute the SDP value
        return sdp_value # replaced by root_heuristic, so no primal needed

    def root_heuristic(self, L, *args, **kwargs) -> npt.NDArray[np.int32]:
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

### Override only SDP (keep default heuristic)

Must call `set_sdp_primal_solution` before returning.

```python
class CustomSDPSolver(QUBOSolver):
    def sdp_bound(self, node, P0, P, *args, **kwargs) -> float:
        # Compute the SDP value and the SDP primal solution matrix
        X = np.identity(P.n) # must be of shape (P.n, P.n)
        self.set_sdp_primal_solution(X) # set it before leaving the function
        return sdp_value
```

---

## Quick Reference: When do I need `set_sdp_primal_solution`?

| Overrides | Need to call `set_sdp_primal_solution`? |
|---|---|
| `sdp_bound` only | Yes, inside `sdp_bound` |
| `root_sdp_bound` only | Yes, inside `root_sdp_bound` |
| `sdp_bound` + `heuristic` | No |
| `sdp_bound` + `root_heuristic` | Yes, inside `sdp_bound` |
| `root_sdp_bound` + `root_heuristic` | No (but non-root default heuristic still needs it via default `sdp_bound`) |
| All four | No |