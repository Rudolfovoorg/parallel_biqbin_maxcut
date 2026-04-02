# in your biqbin Python package, e.g. biqbin/parameters.py

from dataclasses import dataclass
import tomllib
from biqbin.biqbin_module import _Parameters

_FIELDS = [
    ("init_bundle_iter",  int),
    ("max_bundle_iter",   int),
    ("triag_iter",        int),
    ("pent_iter",         int),
    ("hept_iter",         int),
    ("max_outer_iter",    int),
    ("extra_iter",        int),
    ("violated_TriIneq",  float),
    ("TriIneq",           int),
    ("adjust_TriIneq",    int),
    ("PentIneq",          int),
    ("HeptaIneq",         int),
    ("Pent_Trials",       int),
    ("Hepta_Trials",      int),
    ("include_Pent",      int),
    ("include_Hepta",     int),
    ("root",              int),
    ("use_diff",          int),
    ("time_limit",        int),
    ("branchingStrategy", int),
]


class BiqbinParameters(_Parameters):
    def __init__(self, **kwargs):
        super().__init__()  # sets C defaults
        valid = {name for name, _ in _FIELDS}
        for key, value in kwargs.items():
            if key not in valid:
                raise ValueError(f"Unknown parameter: '{key}'")
            setattr(self, key, value)

    def __repr__(self) -> str:
        fields = ", ".join(
            f"{name}={getattr(self, name)!r}" for name, _ in _FIELDS)
        return f"Parameters({fields})"

    def __str__(self) -> str:
        lines = [f"  {name}: {getattr(self, name)}" for name, _ in _FIELDS]
        return "Parameters(\n" + "\n".join(lines) + "\n)"

    def __eq__(self, other) -> bool:
        if not isinstance(other, _Parameters):
            return NotImplemented
        return all(getattr(self, n) == getattr(other, n) for n, _ in _FIELDS)

    def __copy__(self) -> "BiqbinParameters":
        new = BiqbinParameters()
        for name, _ in _FIELDS:
            setattr(new, name, getattr(self, name))
        return new

    @staticmethod
    def from_toml(path: str) -> "BiqbinParameters":
        params = BiqbinParameters()
        with open(path, "rb") as f:
            data = tomllib.load(f)['BiqbinParameters']
        valid = {name for name, _ in _FIELDS}
        for key, value in data.items():
            if key not in valid:
                raise ValueError(f"Unknown parameter: '{key}'")
            setattr(params, key, value)
        return params
