# Utilities

Selected helpers for preparing problem data, exporting QUBOs, and inspecting
results. Import these functions from `biqbin.utils`. Decorators and internal
data-collection helpers are intentionally omitted.

## Matrix conversion and validation

`to_sparse` and `from_sparse` convert between dense matrices and the sparse
**dictionary format** used by BiqBin JSON files. They do not return SciPy sparse
matrix objects.

::: biqbin.utils.to_sparse

::: biqbin.utils.from_sparse

::: biqbin.utils.check_matrix_validity

### Integer scaling

`divide_matrix_by_gcd` modifies the supplied matrix **in place** when its GCD
is greater than one. Pass a copy if you need to preserve the original values.
It expects integer-valued input; validate the matrix first if necessary.
For an all-zero matrix it returns zero and leaves the matrix unchanged.

::: biqbin.utils.divide_matrix_by_gcd

## QUBO preparation and export

For example, prepare a JSON-serializable problem without constructing a solver:

```python
import json
import numpy as np
from biqbin.utils import qubo_to_biqbin_representation

Q = np.array([[1, 2], [0, -1]])
problem_data = qubo_to_biqbin_representation(Q, offset=3, minimization=True)
problem_json = json.dumps(problem_data)
```

For existing `ProblemQubo` objects, use the two `save_qubo_problem_as_*`
functions below. Both overwrite the destination file if it already exists.

::: biqbin.utils.qubo_to_biqbin_representation

::: biqbin.utils.qubo_to_qplib_str

::: biqbin.utils.save_qubo_problem_as_json

::: biqbin.utils.save_qubo_problem_as_qplib

## JSON encoding

Use this callback when serializing structures containing NumPy arrays or
scalars, for example `json.dumps(result, default=convert_numpy_to_json_serializable)`.

::: biqbin.utils.convert_numpy_to_json_serializable

## Result-data analysis

These helpers flatten nested JSON records for tabular analysis. `flatten_dict`
yields key/value pairs; `data_reader_pd` yields dictionaries that can be passed
to `pandas.DataFrame`. Neither helper creates a DataFrame or imports pandas.

::: biqbin.utils.flatten_dict

::: biqbin.utils.data_reader_pd
