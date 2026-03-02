from argparse import ArgumentParser, ArgumentTypeError, Action
from biqbin.biqbin_base import logging, logger
from biqbin.data_parsers import (FromFile,
                                 MaxCutFromJson,
                                 MaxCutFromMatrixMarket,
                                 MaxCutFromEdgeWeights,
                                 QuboFromJson,
                                 QuboFromMatrixMarket,
                                 QuboFromEdgeWeights,
                                 QuboFromQPLIB)


class ArgParserBase(ArgumentParser):
    def __init__(self, prog: str, description: str):
        super().__init__(prog=prog, description=description,
                         usage=f'mpirun [-n N] python3 {prog} problem_instance [-p PARAMS] [-w] [-o OUTPUT]',
                         epilog='For more information please visit https://github.com/Rudolfovoorg/parallel_biqbin_maxcut',
                         )
        self.add_argument('problem_instance',
                          help='Path to the problem instance file')

        # Optional arguments
        self.add_argument('-s', '--solution',
                          help='file path to an initial solution')

        self.add_argument('-c', '--collect-heur-data', action='store_true',
                          help='collect heuristic data on root node (time taken and value)')

        self.add_argument('-p', '--params', default='params',
                          help='custom parameters file path (default: "params")')

        self.add_argument('-w', '--overwrite',
                          action='store_true',
                          help='overwrite output.json instead of labeling with _NUMBER'
                          )
        self.add_argument('-O', '--optimize', action='store_true',
                          help='Divides the final input matrix values by their GCD')
        self.add_argument('-o', '--output', help='set custom output file path')
        # time limit format taken from SLURM docs https://slurm.schedmd.com/sbatch.html
        self.add_argument('-t', '--time', default='0', type=self.parse_time_limit,
                          help='set running time limit; acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"')

        self.add_argument('-v', '--verbose', action='count', default=0,
                          help='Increases logging level from WARNING to -v for INFO or -vv for DEBUG')

    def parse_args(self, *args, **kwargs):
        ns = super().parse_args(*args, **kwargs)

        if ns.verbose == 1:
            logger.setLevel(logging.INFO)
        elif ns.verbose > 1:
            logger.setLevel(logging.DEBUG)
        return ns

    def parse_time_limit(self, s: str) -> int:
        """
        Parse Slurm-style time limits:
        - "MM" (minutes only)
        - "HH:MM:SS"
        - "D-HH:MM:SS"
        Returns:
            int: total seconds
        """
        # If format includes days
        if "-" in s:
            days_str, rest = s.split("-", 1)
            days = int(days_str)
        else:
            days, rest = 0, s

        parts = rest.split(":")
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
        elif len(parts) == 2:
            hours, minutes = map(int, parts)
            seconds = 0
        elif len(parts) == 1:
            # Slurm allows just minutes like "30"
            return int(parts[0]) * 60
        else:
            raise ArgumentTypeError(f"Invalid time format: {s}")

        total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
        return int(total_seconds)


class ArgParserMaxCut(ArgParserBase):

    FORMAT_CHOICES = {
        'json': MaxCutFromJson,
        'mm': MaxCutFromMatrixMarket,
        'ew': MaxCutFromEdgeWeights
    }

    def __init__(self):
        super().__init__(prog='biqbin_maxcut.py', description='Biqbin Maxcut solver')
        self.add_argument(
            '--format',
            default=self.FORMAT_CHOICES["json"],
            type=self.parse_format,
            help='Max-Cut problem instance file format. mm is MatrixMarket, ew is edge-weight Stanford GSet style')

    def parse_format(self, fmt: str) -> FromFile:
        fmt = fmt.lower()
        if fmt not in self.FORMAT_CHOICES:
            raise ArgumentTypeError(
                f'invalid format: {fmt}, choose from {tuple(i for i in self.FORMAT_CHOICES.keys())}')

        return self.FORMAT_CHOICES[fmt]


class ArgParserQubo(ArgParserBase):

    FORMAT_CHOICES = {
        'json': QuboFromJson,
        'mm': QuboFromMatrixMarket,
        'ew': QuboFromEdgeWeights,
        'qplib': QuboFromQPLIB
    }

    def __init__(self, prog='biqbin_qubo.py', description='Biqbin QUBO solver'):
        super().__init__(prog=prog, description=description)
        self.add_argument(
            '--format',
            choices=self.FORMAT_CHOICES.values(),
            default=self.FORMAT_CHOICES["json"],
            type=self.parse_format,
            help='QUBO problem instance file format. mm is MatrixMarket, ew is edge-weight Stanford GSet style')

    def parse_format(self, fmt: str) -> FromFile:
        fmt = fmt.lower()
        if fmt not in self.FORMAT_CHOICES:
            raise ArgumentTypeError(
                f'invalid format: {fmt}, choose from {tuple(i for i in self.FORMAT_CHOICES.keys())}')

        return self.FORMAT_CHOICES[fmt]


class ArgParserDWaveHeuristic(ArgParserQubo):
    def __init__(self):
        super().__init__(prog='biqbin_heuristic.py',
                         description='Biqbin QUBO solver with DWave heuristic')
