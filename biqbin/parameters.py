from pathlib import Path
import tomllib
from biqbin.biqbin_module import _Parameters


class BiqbinParameters(_Parameters):
    """Detailed explanation of all parameters is in docs/PARAMETERS.md."""

    def __init__(self, **kwargs):
        super().__init__()  # sets C defaults
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ValueError(f'Unknown parameter: "{key}"')
            setattr(self, key, value)

    def __repr__(self) -> str:
        fields = ", ".join(
            f"{name}={getattr(self, name)!r}"
            for name in dir(self)
            if not name.startswith("_") and not callable(getattr(self, name))
        )
        return f"BiqbinParameters({fields})"

    def __str__(self) -> str:
        lines = [
            f"  {name}: {getattr(self, name)}"
            for name in dir(self)
            if not name.startswith("_") and not callable(getattr(self, name))
        ]
        return "BiqbinParameters(\n" + "\n".join(lines) + "\n)"

    @staticmethod
    def from_toml(path: str | Path) -> "BiqbinParameters":
        with open(path, "rb") as f:
            data = tomllib.load(f).get("BiqbinParameters")
        if data is None:
            raise ValueError(
                "TOML file must contain a [BiqbinParameters] section")

        return BiqbinParameters(**data)
