"""Custom QUBO heuristic using the shared runtime and class-owned options."""

import numpy as np
from numpy import typing as npt

from biqbin import QUBOSolver
from biqbin.cli import main
from biqbin.configuration import Option


class QuboDwaveSampler(QUBOSolver):
    @classmethod
    def config_options(cls):
        return {
            **super().config_options(),
            "num_reads": Option(
                int,
                default=10,
                minimum=1,
                help="Number of simulated-annealing reads",
                aliases=("--num-reads",),
            ),
        }

    def __init__(
        self,
        problem,
        params=None,
        time_limit=None,
        initial_estimate=None,
        collect_heuristic_root_data=False,
        collect_sdp_bound_root_data=False,
        sampler=None,
        *,
        parameters=None,
        num_reads=None,
        **sampler_kwargs,
    ):
        option = self.config_options()["num_reads"]
        num_reads = option.convert(option.default if num_reads is None else num_reads, "num_reads")
        super().__init__(
            problem,
            params,
            time_limit,
            initial_estimate,
            collect_heuristic_root_data,
            collect_sdp_bound_root_data,
            parameters=parameters,
        )
        if sampler is None:
            from dwave.samplers import SimulatedAnnealingSampler

            sampler = SimulatedAnnealingSampler()
        self.sampler = sampler
        self.sampler_kwargs = {"num_reads": num_reads, **sampler_kwargs}

    def heuristic(self, L: np.ndarray, **kwargs) -> npt.ArrayLike:
        """Heuristc with D-Waves simulated annealing sampler

        Args:
            L (np.ndarray): Subproblem Laplacean matrix

        Returns:
            np.ndarray: Solution [0, 1] binary vector of size L.shape[0] - 1
        """
        x = np.array(
            list(
                self.sampler.sample_qubo(-L[:-1, :-1], **self.sampler_kwargs).first.sample.values()
            ),
            dtype=np.int32,
        )
        return x


if __name__ == "__main__":
    raise SystemExit(main(kind="qubo", solver_factory=QuboDwaveSampler))
