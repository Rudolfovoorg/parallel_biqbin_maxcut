## Biqbin custom heuristic guide

Biqbin allows running a custom heuristic function on all B&B nodes or injecting a precomputed solution estimate to use one the root node.

## Overriding heuristic

Both `MaxCutSolver` and `QUBOSolver` have a overridable `heuristic` method with the following signature:

```py
class NewHeuristicSolver(QUBOSolver): # or NewHeuristicSolver(MaxCutSolver)

    @heur_root_data_collector()
    def heuristic(self, L0: np.ndarray, L: np.ndarray, xfixed: np.ndarray, sol_X: np.ndarray, x: np.ndarray) -> float:
        # Run some heuristic
        # ...
        # heuristic_solution_found = [0, 1, ..., 0, 0]

        # Copy the solution into x where xfixed == 0
        free_mask = (xfixed == 0)
        np.copyto(x, sol_X)
        x[free_mask] = heuristic_solution_found
        
        return self._evaluate_solution(L0, x)
```

- `@heur_root_data_collector` stores the objective value and compute time of each call to the `heuristic` method on the **root** node.

- `self._evaluate_solution(L0, x)` helper function that evaluates the solution found by the heuristic

Arguments:

- `L0` (`np.ndarray`): original Problem \*SP->L matrix. Laplacean matrix for the MaxCut problem
- `L` (`np.ndarray`): subproblem Problem \*PP->L matrix.
- `xfixed` (`np.ndarray`): Binary vector where 1 means a variable is fixed. Size of subproblem L is smaller by the sum of xfixed.
- `sol_X` (`np.ndarray`): Binary vector where the values of fixed variables are stored.

Argument used by Biqbin:

- `x` (`np.ndarray`): stores the solution of the heuristic function, used by the solver to determine the MaxCut lower bound.

Output:

- `float`: Objective value of the heuristic solution `x`.

## Passing in initial estimate solution

Solvers can be provided with an estimate solution on initialization which will be used on the root node instead of a heuristic solution.

```py
# biqbin_qubo.py
# ...

solver = QUBOSolver(problem=problem,
                    params=args.params,
                    time_limit=args.time,
                    initial_estimate=initial_estimate, # Initial estimate as a binary np.ndarray
                    collect_heur_data=args.collect_heur_data
                    )
```

Running any biqbin_x.py files with the `-s path_to_estimate_solution` arguments will load in an estimate solution from a json dictionary with a `estimate_solution` key and a list of 1 and 0 as value. 

Example QUBO:
```bash
mpirun python3 biqbin_qubo.py tests/qubos/40/kcluster40_025_10_1.json -s tests/w_solution/kcluster40_025_10_1.json_initial_solution.json
```

Example Max-Cut:
```bash
mpirun python3 biqbin_maxcut.py tests/rudy/g05_60.0.json -s tests/w_solution/g05_60.0.json_initial_solution.json
```

> **Note** problem instance and estimate solution do not need to be in separate files, the .json can contain both `qubo`/`maxcut` and `estimate_solution` keys at the top level and can be passed in like:
>  ```mpirun python3 biqbin_qubo.py problem_instance.json -s problem_instance.json```
