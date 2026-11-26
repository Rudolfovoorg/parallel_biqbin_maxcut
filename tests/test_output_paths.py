"""Preserve existing output files, including gaps and competing writers."""

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, local
from types import SimpleNamespace

import pytest

from biqbin.data_parsers import BQPToJson, MaxCutSolutionToJson, QuboSolutionToJson, ToFile


@pytest.mark.parametrize(
    "writer,section",
    [(MaxCutSolutionToJson, "maxcut"), (QuboSolutionToJson, "qubo"), (BQPToJson, "bqp")],
)
def test_writers_preserve_numbering_gaps(tmp_path, writer, section):
    filename = tmp_path / "result.json"
    filename.write_text("original")
    (tmp_path / "result_2.json").write_text("keep this result")
    solution = SimpleNamespace(solution={"computed_val": 42}, meta_data={})
    writer(solution).write(str(filename))
    assert filename.read_text() == "original"
    assert (tmp_path / "result_2.json").read_text() == "keep this result"
    assert json.loads((tmp_path / "result_1.json").read_text())[section]["computed_val"] == 42


def test_names_with_glob_characters_are_literal(tmp_path):
    filename = tmp_path / "result[1].json"
    filename.write_text("original")
    assert ToFile.get_output_path(str(filename), False) == str(tmp_path / "result[1]_1.json")
    assert ToFile.get_output_path(str(filename), True) == str(filename)


def test_dangling_symlink_is_not_overwritten(tmp_path):
    filename = tmp_path / "result.json"
    filename.symlink_to(tmp_path / "missing")
    with ToFile.open_output(str(filename), False) as output:
        output.write("new result")
    assert filename.is_symlink()
    assert not (tmp_path / "missing").exists()
    assert (tmp_path / "result_1.json").read_text() == "new result"


def test_competing_writers_claim_distinct_files(tmp_path, monkeypatch):
    # Force every writer to select the same initial path before any can open it.
    workers = 4
    barrier = Barrier(workers)
    state = local()
    original = ToFile.get_output_path

    def select_path(cls, filename, overwrite):
        path = original(filename, overwrite)
        if not getattr(state, "selected", False):
            state.selected = True
            barrier.wait(timeout=10)
        return path

    monkeypatch.setattr(ToFile, "get_output_path", classmethod(select_path))

    def write(value):
        with ToFile.open_output(str(tmp_path / "result.json"), False) as output:
            json.dump({"value": value}, output)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(write, range(workers)))
    outputs = list(tmp_path.glob("*.json"))
    assert len(outputs) == workers
    assert {json.loads(path.read_text())["value"] for path in outputs} == set(range(workers))
