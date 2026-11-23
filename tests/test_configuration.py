"""Configuration precedence, validation and component-owned output behavior."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from biqbin.configuration import Option, load_config, load_legacy_parameters, resolve_options
from biqbin.data_parsers import (
    BQPToJson,
    FromFile,
    MaxCutSolutionToJson,
    QuboSolutionToJson,
    ToFile,
)
from biqbin.problem import ProblemBQP, ProblemMaxCut, ProblemQubo


def test_explicit_false_and_zero_override_file_values_without_changing_definitions():
    definitions = {
        "enabled": Option(bool, default=True),
        "time_limit": Option(int, default=10, minimum=0),
    }
    values = {"enabled": True, "time_limit": 600}
    assert resolve_options(definitions, values, {"enabled": False, "time_limit": 0}) == {
        "enabled": False,
        "time_limit": 0,
    }
    assert resolve_options(definitions, values) == values
    assert resolve_options(definitions) == {"enabled": True, "time_limit": 10}


@pytest.mark.parametrize(
    "option,value,expected",
    [
        (Option(bool), "false", False),
        (Option(bool), "TRUE", True),
        (Option(int), "0", 0),
        (Option(float), "1e-3", 0.001),
        (Option(str), Path("input.json"), "input.json"),
        (Option(str), None, None),
    ],
)
def test_option_converts_cli_values(option, value, expected):
    assert option.convert(value) == expected


@pytest.mark.parametrize(
    "option,value",
    [
        (Option(bool), "perhaps"),
        (Option(bool), 1),
        (Option(int), True),
        (Option(int), 1.5),
        (Option(int), "1.5"),
        (Option(float), True),
        (Option(float), "nan"),
        (Option(float), float("inf")),
        (Option(str), 12),
        (Option(str, choices=("json", "mm")), "yaml"),
        (Option(int, minimum=0), -1),
    ],
)
def test_invalid_option_has_its_name_in_error(option, value):
    with pytest.raises(ValueError, match="setting:"):
        option.convert(value, "setting")


def test_missing_and_unknown_options_are_errors():
    options = {"path": Option(str, required=True)}
    with pytest.raises(ValueError, match="Missing required option: path"):
        resolve_options(options)
    with pytest.raises(ValueError, match="Unknown configuration options: typo"):
        resolve_options(options, {"typo": "x"})
    with pytest.raises(ValueError, match="Unknown configuration options: typo"):
        resolve_options(options, {"path": "input"}, {"typo": "x"})
    with pytest.raises(ValueError, match="mapping"):
        resolve_options(options, "input")


def test_load_toml_and_validate_component_values(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text('[reader]\npath = "input.json"\n[problem]\noptimize = false\n')
    settings = load_config(path)
    assert resolve_options(FromFile.config_options(), settings["reader"]) == {"path": "input.json"}
    for cls in (ProblemMaxCut, ProblemQubo, ProblemBQP):
        assert resolve_options(cls.config_options(), settings["problem"]) == {"optimize": False}
    assert "optimize" not in FromFile.config_options()


def test_toml_error_identifies_file(tmp_path):
    path = tmp_path / "broken.toml"
    path.write_text("[broken")
    with pytest.raises(ValueError, match="broken.toml"):
        load_config(path)


def test_legacy_file_comments_and_numbers(tmp_path):
    path = tmp_path / "params"
    path.write_text("# comment\n\nmax_bundle_iter = 20 # override\nviolated_TriIneq = 1e-3\n")
    assert load_legacy_parameters(path) == {"max_bundle_iter": 20, "violated_TriIneq": 0.001}


@pytest.mark.parametrize(
    "content",
    ["invalid", "x =", "= 3", "x = 1 = 2", "x = nope", "x = nan", "x = inf", "x = 1\nx = 2"],
)
def test_invalid_legacy_file_has_line_number(tmp_path, content):
    path = tmp_path / "params"
    path.write_text(content)
    with pytest.raises(ValueError, match=r"params:\d+:"):
        load_legacy_parameters(path)


def test_numbered_output_uses_free_path_with_gaps_and_literal_brackets(tmp_path):
    output = tmp_path / "result[1].json"
    output.write_text("original")
    occupied = tmp_path / "result[1]_2.json"
    occupied.write_text("keep")
    unrelated = tmp_path / "result1_1.json"
    unrelated.write_text("also keep")
    assert ToFile.get_output_path(str(output), False) == str(tmp_path / "result[1]_1.json")
    assert ToFile.get_output_path(str(output), True) == str(output)
    assert output.read_text() == "original"
    assert occupied.read_text() == "keep"


@pytest.mark.parametrize(
    "writer_cls,section",
    [(MaxCutSolutionToJson, "maxcut"), (QuboSolutionToJson, "qubo"), (BQPToJson, "bqp")],
)
def test_writers_own_options_and_return_actual_path(tmp_path, writer_cls, section):
    solution = SimpleNamespace(
        solution={"computed_val": 4},
        meta_data={"test": True},
        solution_maxcut={"computed_val": 5},
        maxcut_solution={"computed_val": 5},
    )
    options = resolve_options(writer_cls.config_options(), {"path": tmp_path / "result.json"})
    output = options.pop("path")
    writer = writer_cls(solution)
    first = writer.write(output, **options)
    second = writer.write(output, **options)
    assert first == output
    assert second == str(tmp_path / "result_1.json")
    data = json.loads(Path(second).read_text())
    assert data[section] == solution.solution
    assert data["meta_data"] == solution.meta_data
    if section != "maxcut":
        assert data["maxcut"] == {"computed_val": 5}
