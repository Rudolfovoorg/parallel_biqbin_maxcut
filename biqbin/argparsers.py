"""Compatibility namespaces backed by the shared CLI and configuration resolver.

New applications should call ``biqbin.cli.main`` or ``biqbin.runtime.run``.
These adapters preserve the old attribute names and logging side effect.
"""

from argparse import ArgumentParser, ArgumentTypeError

from biqbin.cli import cli_overrides, configure_parser, parse_time_limit
from biqbin.logging import configure_verbosity
from biqbin.runtime import COMPONENTS, resolve_config


class ArgParserBase(ArgumentParser):
    parse_time_limit = staticmethod(parse_time_limit)

    def __init__(self, prog, description, *, kind):
        super().__init__(prog=prog, description=description)
        self.kind = kind
        configure_parser(self, kind)

    def parse_format(self, value):
        try:
            return self.FORMAT_CHOICES[value.lower()]
        except KeyError:
            raise ArgumentTypeError(
                f"Invalid format {value!r}; choose from {tuple(self.FORMAT_CHOICES)}"
            ) from None

    def parse_args(self, *args, **kwargs):
        namespace = super().parse_args(*args, **kwargs)
        try:
            config = resolve_config(
                config_file=namespace.config,
                params_file=namespace.params,
                overrides=cli_overrides(namespace, self.kind),
            )
        except (ValueError, OSError, TypeError) as error:
            self.error(str(error))
        namespace.problem_instance = config["reader"]["path"]
        namespace.format = self.FORMAT_CHOICES[config["format"]]
        namespace.params = config["solver"]
        namespace.time = config["solver"]["time_limit"]
        namespace.optimize = config["problem"]["optimize"]
        namespace.output = config["writer"]["path"]
        namespace.overwrite = config["writer"]["overwrite"]
        namespace.solution = config["run"]["initial_solution"]
        namespace.collect_root_data = config["run"]["collect_root_data"]
        namespace.verbose = config["run"]["verbose"]
        namespace.log_dir = config["run"]["log_dir"]
        configure_verbosity(namespace.verbose)
        return namespace


class ArgParserMaxCut(ArgParserBase):
    FORMAT_CHOICES = COMPONENTS["maxcut"][3]

    def __init__(self):
        super().__init__("biqbin_maxcut.py", "BiqBin Max-Cut solver", kind="maxcut")


class ArgParserQubo(ArgParserBase):
    FORMAT_CHOICES = COMPONENTS["qubo"][3]

    def __init__(self, prog="biqbin_qubo.py", description="BiqBin QUBO solver"):
        super().__init__(prog, description, kind="qubo")


class ArgParserDWaveHeuristic(ArgParserQubo):
    def __init__(self):
        super().__init__("biqbin_heuristic.py", "BiqBin QUBO solver with custom heuristic")


class ArgParserBQP(ArgParserBase):
    FORMAT_CHOICES = COMPONENTS["bqp"][3]

    def __init__(self):
        super().__init__("biqbin_bqp.py", "BiqBin BQP solver", kind="bqp")
