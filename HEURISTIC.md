## Biqbin custom heuristic guide


Biqbin allows running a custom heuristic function on all B&B nodes or injecting a precomputed solution estimate to use one the root node.

## Overriding heuristic

Both `MaxCutSolver` and `QuboSolver` have a overridable `heuristic` method with the following signature:

```py
    @heur_root_data_collector()
    def heuristic(self, L0: np.ndarray, L: np.ndarray, xfixed: np.ndarray, sol_X: np.ndarray, x: np.ndarray) -> float:
        ...
        return objective_value_found_by_heuristic_function
```
- `@heur_root_data_collector` stores the objective value and compute time of each call to the `heuristic` method on the **root** node.

```py

```