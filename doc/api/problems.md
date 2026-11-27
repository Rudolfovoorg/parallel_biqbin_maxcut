# Problems

Construct a problem object from your matrices and vectors, then pass it to the
matching [solver](solvers.md). Constructor signatures and descriptions below
come from the Python source.

::: biqbin.ProblemMaxCut
    options:
      members:
        - maxcut_adjacency_matrix

::: biqbin.ProblemQubo
    options:
      members:
        - Q
        - offset
        - is_minimization
        - maxcut_adjacency_matrix

::: biqbin.ProblemBQP
    options:
      members:
        - F
        - c
        - A
        - b
        - offset
        - maxcut_adjacency_matrix
