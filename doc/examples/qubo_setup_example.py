"""Solve a small QUBO and export its input; run as a module under MPI."""

import json

import numpy as np

from biqbin import ProblemQubo, QUBOSolver, init
from biqbin.utils import qubo_to_biqbin_representation


def main():
    init()  # Every MPI rank must initialize and participate in compute().
    qubo = np.array([[-1, 3], [3, -1]])
    offset = 10
    problem = ProblemQubo(
        Q=qubo,
        offset=offset,
        problem_name="example_qubo",
        is_minimization=True,
    )
    result = QUBOSolver(problem).compute()

    # Only rank 0 receives a result. Workers receive None.
    if result is not None:
        print("Objective:", result.solution["computed_val"])
        print("Binary solution:", result.solution["x"])

        # Export the input for reuse with biqbin_qubo.py (overwrites this file).
        qubo_dict = qubo_to_biqbin_representation(qubo=qubo, offset=offset)
        with open("example_qubo.json", "w") as f:
            json.dump(qubo_dict, f, indent=2)
        print("Input saved to example_qubo.json")


if __name__ == "__main__":
    main()
