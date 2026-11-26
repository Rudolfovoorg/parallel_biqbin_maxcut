"""Run inside the runtime image as its default nonroot user; no pytest required."""

import json
import os
from pathlib import Path

import numpy as np

from biqbin import ProblemQubo, QuboSolutionToJson, QUBOSolver, init


def main():
    if os.geteuid() == 0:
        raise RuntimeError("The runtime image must default to a nonroot user")
    if "OMPI_ALLOW_RUN_AS_ROOT" in os.environ:
        raise RuntimeError("Runtime smoke must not inherit CI root overrides")

    init()
    problem = ProblemQubo(np.diag([-2, 3]), 0, "runtime_smoke", True)
    result = QUBOSolver(problem).compute()
    if result is not None:
        assert result.solution["computed_val"] == -2
        assert result.solution["x"] == [1, 0]
        # Check both bundled-example and data-directory output locations.
        for directory in (Path("/solver"), Path("/data")):
            path = directory / "runtime-smoke.json"
            QuboSolutionToJson(result).write(str(path))
            assert json.loads(path.read_text())["qubo"]["computed_val"] == -2
        print("Runtime MPI solve and output writes passed", flush=True)


if __name__ == "__main__":
    main()
