# Solutions

Solver `compute()` methods return a solution object on MPI rank 0 and `None`
on worker ranks. Use `solution` for the problem-specific result and
`meta_data` for run statistics. These objects are normally created by the
solver rather than constructed directly by application code.

::: biqbin.SolutionMaxCut
    options:
      members:
        - solution
        - problem
        - meta_data
        - format_summary
        - print_computed_solution

::: biqbin.SolutionQubo
    options:
      members:
        - solution
        - problem
        - meta_data
        - format_summary
        - print_computed_solution

::: biqbin_bqp.SolutionBQP
    options:
      members:
        - solution
        - problem
        - meta_data
        - format_summary
        - print_computed_solution
