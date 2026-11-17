# Solvers

Call [biqbin.init()](entrypoints.md#biqbin.init) before constructing a solver.
Every MPI rank participates in `compute()`; only rank 0 receives the result.

The reference includes the public callback methods used to customize bounds
and heuristics. See the [custom solver guide](../CUSTOM_SOLVER.md) and
[callback guide](../CUSTOM_SDP_HEURISTIC.md) for their contracts and examples.

::: biqbin.MaxCutSolver
    options:
      members:
        - problem
        - compute
        - initial_sdp_bound
        - sdp_bound
        - initial_heuristic
        - heuristic
        - set_sdp_primal_solution

::: biqbin.QUBOSolver
    options:
      members:
        - problem
        - compute
        - initial_sdp_bound
        - sdp_bound
        - initial_heuristic
        - heuristic
        - set_sdp_primal_solution

::: biqbin_bqp.BQPSolver
    options:
      members:
        - problem
        - compute
        - initial_sdp_bound
        - sdp_bound
        - initial_heuristic
        - heuristic
        - set_sdp_primal_solution
