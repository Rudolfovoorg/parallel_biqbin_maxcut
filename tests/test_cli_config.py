"""Shared CLI definitions, precedence and compatibility adapters."""

import argparse

import pytest

from biqbin.argparsers import ArgParserQubo
from biqbin.cli import cli_overrides, configure_parser, parse_time_limit
from biqbin.runtime import resolve_config
from biqbin_heuristic import QuboDwaveSampler


@pytest.mark.parametrize(
    "flags",
    [
        ["--param", "time_limit=60", "-t", "0"],
        ["-t", "0", "--param", "time_limit=60"],
        ["--solver-time-limit", "60", "-t", "0"],
        ["--param", "max_bundle_iter=20", "--param", "max_bundle_iter=25"],
    ],
)
def test_duplicate_solver_settings_rejected(flags, capsys):
    parser = configure_parser(argparse.ArgumentParser(), "qubo")
    with pytest.raises(SystemExit):
        parser.parse_args(["input.json", *flags])
    assert "supplied more than once" in capsys.readouterr().err


def test_configuration_file_formats_are_exclusive(tmp_path):
    parser = configure_parser(argparse.ArgumentParser(), "qubo")
    with pytest.raises(SystemExit):
        parser.parse_args(["input.json", "--config", "a.toml", "-p", "params"])
    with pytest.raises(ValueError, match="not both"):
        resolve_config(config_file="a.toml", params_file="params")


def test_legacy_adapter_preserves_file_time_limit_and_custom_flags(tmp_path):
    class CustomParser(ArgParserQubo):
        def __init__(self):
            super().__init__()
            self.add_argument("--test-case", type=int)

    params = tmp_path / "params"
    params.write_text("time_limit=60\n")
    args = CustomParser().parse_args(["input.json", "-p", str(params), "--test-case", "4"])
    assert args.params["time_limit"] == args.time == 60
    assert args.test_case == 4
    args = ArgParserQubo().parse_args(["input.json", "-p", str(params), "-t", "0"])
    assert args.params["time_limit"] == args.time == 0


@pytest.mark.parametrize("value", ["-1", "1:-2", "bad", "1:2:3:4"])
def test_shared_time_parser_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_time_limit(value)


def test_custom_option_inherits_native_settings_and_uses_file_then_cli(tmp_path):
    config = tmp_path / "run.toml"
    config.write_text('kind="qubo"\n[reader]\npath="input.json"\n[solver]\nnum_reads=12\n')
    parser = configure_parser(argparse.ArgumentParser(), "qubo", QuboDwaveSampler)
    args = parser.parse_args(["--config", str(config), "--num-reads", "21"])
    resolved = resolve_config(
        config_file=args.config,
        overrides=cli_overrides(args, "qubo"),
        solver_factory=QuboDwaveSampler,
    )
    assert resolved["solver"]["num_reads"] == 21
    assert resolved["solver"]["init_bundle_iter"] == 5
    assert "num_reads" not in resolve_config({"kind": "qubo", "reader": {"path": "x"}})["solver"]
    with pytest.raises(ValueError, match="num_reads"):
        resolve_config(
            {"kind": "qubo", "reader": {"path": "x"}, "solver": {"num_reads": 0}},
            solver_factory=QuboDwaveSampler,
        )
