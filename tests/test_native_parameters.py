"""Validate the actual optimized extension and standalone native reader."""

import json
import shutil

import pytest

from biqbin.biqbin_module import parameter_schema, validate_parameters
from biqbin.configuration import load_legacy_parameters
from tests.mpi_helpers import ROOT, run_mpi


def test_native_defaults_match_shipped_profile():
    defaults = {name: spec["default"] for name, spec in parameter_schema().items()}
    assert defaults == load_legacy_parameters(ROOT / "params")
    assert validate_parameters({}) == defaults
    assert validate_parameters({"time_limit": 0})["time_limit"] == 0


@pytest.mark.parametrize(
    "values",
    [
        {"violated_TriIneq": float("nan")},
        {"violated_TriIneq": float("inf")},
        {"violated_TriIneq": float("-inf")},
        {"time_limit": 2**80},
        {"time_limit": -1},
        {"time_limit": True},
        {"time_limit": 1.5},
        {"time_limit": "60"},
        {"branchingStrategy": 2},
        {"root": 2},
        {"root": -1},
        {"init_bundle_iter": 0},
        {"init_bundle_iter": 20, "max_bundle_iter": 10},
        {"TriIneq": 0},
        {"TriIneq": 50001},
        {"max_outer_iter": 2**31 - 1},
        {"unknown": 1},
    ],
)
def test_native_parameter_validation(values):
    with pytest.raises(ValueError):
        validate_parameters(values)


def test_native_parameter_boundaries():
    values = {
        "time_limit": 2**31 - 1,
        "TriIneq": 50000,
        "branchingStrategy": 0,
        "max_outer_iter": 2**31 - 11,
        "extra_iter": 10,
    }
    result = validate_parameters(values)
    assert all(result[name] == value for name, value in values.items())


@pytest.mark.parametrize(
    "text",
    [
        "violated_TriIneq=nan\n",
        "violated_TriIneq=inf\n",
        "violated_TriIneq=-inf\n",
        "time_limit=60\ntime_limit=0\n",
        "time_limit=60.0\n",
        "time_limit=1e2\n",
        "time_limit=999999999999999999999\n",
        "unknown=1\n",
        "root=2\n",
        "time_limit=1 garbage\n",
    ],
)
def test_native_and_python_reject_invalid_parameter_files(tmp_path, text):
    params = tmp_path / "bad.params"
    params.write_text(text)
    with pytest.raises(ValueError):
        validate_parameters(load_legacy_parameters(params))
    instance = tmp_path / "instance"
    shutil.copyfile(ROOT / "tests/rudy/g05_60.0", instance)
    code = f"import os; os.execv({str(ROOT / 'biqbin_executable')!r}, ['biqbin', {str(instance)!r}, {str(params)!r}])"
    run_mpi(tmp_path, "-c", code, success=False)


def test_native_parameters_are_broadcast_from_root(tmp_path):
    script = tmp_path / "broadcast.py"
    script.write_text(
        "import json, numpy as np\n"
        "from biqbin import init, get_rank, ProblemQubo, QUBOSolver\n"
        "from biqbin.biqbin_module import run as native_run\n"
        "init()\n"
        "rank=get_rank()\n"
        "p=ProblemQubo(np.array([[-1,3],[3,-1]]),10,'small',True)\n"
        "s=QUBOSolver(p)\n"
        "matrix=p.maxcut_adjacency_matrix.astype(float) if rank==0 else None\n"
        "r=native_run('test','small',matrix,{'time_limit':120 if rank==0 else 1})\n"
        "open(f'rank-{rank}.json','w').write(json.dumps(r['meta_data']['solver_parameters']))\n"
    )
    run_mpi(tmp_path, str(script))
    for rank in range(3):
        assert json.loads((tmp_path / f"rank-{rank}.json").read_text())["time_limit"] == 120
