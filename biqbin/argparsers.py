from argparse import ArgumentParser, ArgumentTypeError


class ArgParserBase(ArgumentParser):
    def __init__(self, prog: str, description: str):
        super().__init__(prog=prog, description=description,
                         usage=f'mpirun [-n N] python3 {prog} problem_instance [-p PARAMS] [-w] [-o OUTPUT]',
                         epilog='For more information please visit https://github.com/Rudolfovoorg/parallel_biqbin_maxcut',
                         )
        self.add_argument('problem_instance',
                          help='Path to the problem instance file')

        # Optional arguments
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

        self.add_argument('-v', '--verbose', action='store_true',
                          help='Verbose prints to terminal')

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

        total_seconds = days*86400 + hours*3600 + minutes*60 + seconds
        return int(total_seconds)


class ArgParserMaxCut(ArgParserBase):
    def __init__(self):
        super().__init__(prog=f'biqbin_maxcut.py', description='Biqbin Maxcut solver')
        self.add_argument('-e', '--edge_weight',
                          action='store_true', help='use edge weight input file')


class ArgParserQubo(ArgParserBase):
    def __init__(self, prog=f'biqbin_qubo.py', description='Biqbin QUBO solver'):
        super().__init__(prog=prog, description=description)
        # self.add_argument('--qplib', action='store_true',
        #                   help='Use .qplib file format')


class ArgParserDWaveHeuristic(ArgParserQubo):
    def __init__(self):
        super().__init__(prog='biqbin_heuristic.py',
                         description='Biqbin QUBO solver with DWave heuristic')
        self.add_argument('-d', '--debug', action='store_true',
                          help='enable debug logs')
        self.add_argument('-i', '--info', action='store_true',
                          help='enable info logs')
