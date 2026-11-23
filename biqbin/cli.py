"""Command-line adapters for the shared runtime pipeline."""

import argparse
import sys

from biqbin.runtime import COMPONENTS, component_options, run


def parse_time_limit(value):
    """Legacy durations: minutes, HH:MM, HH:MM:SS, or D-HH[:MM[:SS]]."""
    try:
        days, rest = value.split("-", 1) if "-" in value else ("0", value)
        parts = [int(part) for part in rest.split(":")]
        if len(parts) == 1:
            seconds = parts[0] * (3600 if "-" in value else 60)
        elif len(parts) == 2:
            seconds = parts[0] * 3600 + parts[1] * 60
        elif len(parts) == 3:
            seconds = parts[0] * 3600 + parts[1] * 60 + parts[2]
        else:
            raise ValueError
        if int(days) < 0 or any(part < 0 for part in parts):
            raise ValueError
        return int(days) * 86400 + seconds
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(f"Invalid time format: {value}") from None


def configure_parser(parser, kind, solver_factory=None):
    """Build the CLI from the selected classes' option definitions."""
    specs = component_options(kind, solver_factory=solver_factory)
    parser.add_argument("problem_instance", nargs="?", default=argparse.SUPPRESS,
                        help=specs["reader"]["path"].help)
    parser.add_argument("--config", help="TOML run configuration")
    parser.add_argument("-p", "--params", help="Legacy key=value solver parameter file")
    parser.add_argument("--format", choices=tuple(COMPONENTS[kind][3]), default=argparse.SUPPRESS)
    parser.add_argument("--param", action="append", default=[], metavar="NAME=VALUE",
                        help="Override a class-defined solver parameter")
    aliases = {
        ("problem", "optimize"): ["-O", "--optimize"],
        ("writer", "path"): ["-o", "--output"],
        ("writer", "overwrite"): ["-w", "--overwrite"],
        ("writer", "with_metadata"): ["--metadata"],
        ("writer", "with_maxcut_solution"): ["--maxcut-solution"],
        ("run", "initial_solution"): ["-s", "--solution"],
        ("run", "collect_root_data"): ["-c", "--collect-root-data"],
        ("run", "verbose"): ["-v", "--verbose"],
        ("run", "log_dir"): ["--log-dir"],
    }
    for section, options in specs.items():
        for name, option in options.items():
            if section == "reader" and name == "path":
                continue
            flags = option.aliases or aliases.get((section, name), [f"--{section}-{name.replace('_', '-')}" ])
            kwargs = {"dest": f"{section}.{name}", "default": argparse.SUPPRESS, "help": option.help}
            if option.type is bool:
                kwargs["action"] = argparse.BooleanOptionalAction
            elif (section, name) == ("run", "verbose"):
                kwargs["action"] = "count"
            elif (section, name) == ("run", "log_dir"):
                kwargs.update(nargs="?", const="")
            parser.add_argument(*flags, **kwargs)
    parser.add_argument("-t", "--time", dest="solver.time_limit", default=argparse.SUPPRESS,
                        type=parse_time_limit, help=parse_time_limit.__doc__)
    parser.add_argument("--collect-heur-data", dest="run.collect_root_data", action="store_true",
                        default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("-j", "--json", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("-d", "--debug", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("-i", "--info", action="store_true", default=False, help=argparse.SUPPRESS)
    return parser


def cli_overrides(args, kind):
    values = vars(args)
    result = {"kind": kind}
    for name, value in values.items():
        if "." in name:
            section, key = name.split(".", 1)
            result.setdefault(section, {})[key] = value
    if "problem_instance" in values:
        result.setdefault("reader", {})["path"] = values["problem_instance"]
    if "format" in values:
        result["format"] = values["format"]
    if args.json:
        result["format"] = "json"
    if args.debug or args.info:
        options = result.setdefault("run", {})
        options["verbose"] = max(options.get("verbose", 0), 2 if args.debug else 1)
    for setting in args.param:
        key, separator, value = setting.partition("=")
        if not separator or not key:
            raise ValueError("--param expects NAME=VALUE")
        result.setdefault("solver", {})[key] = value
    return result


def main(argv=None, *, kind=None, solver_factory=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if kind is None:
        selector = argparse.ArgumentParser(prog="biqbin", description="BiqBin MPI runtime pipeline", add_help=False)
        selector.add_argument("kind", choices=tuple(COMPONENTS))
        if not argv or argv[0] in {"-h", "--help"}:
            selector.print_help()
            return 0
        selected, argv = selector.parse_known_args(argv)
        kind = selected.kind
    parser = configure_parser(argparse.ArgumentParser(prog=f"biqbin {kind}"), kind, solver_factory)
    args = parser.parse_args(argv)
    try:
        overrides = cli_overrides(args, kind)
        run(config_file=args.config, params_file=args.params, overrides=overrides,
            solver_factory=solver_factory)
    except (ValueError, OSError, TypeError) as exc:
        from biqbin import _initialized, get_rank
        if not _initialized or get_rank() == 0:
            print(f"BiqBin: {exc}", file=sys.stderr)
        return 2
    return 0
