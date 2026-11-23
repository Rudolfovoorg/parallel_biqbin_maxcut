"""Small option definitions shared by runtime configuration and CLI parsing."""

import math
import re
import tomllib
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class Option:
    """Describe one component option without performing any runtime actions.

    Defaults are applied by :func:`resolve_options`, after explicit values have
    been collected. ``None`` is allowed for optional fields with a ``None`` default.
    """

    type: type
    default: Any = None
    required: bool = False
    help: str = ""
    choices: tuple | None = None
    aliases: tuple[str, ...] = ()
    minimum: int | float | None = None

    def convert(self, value: Any, name: str = "value") -> Any:
        """Convert a CLI string or validate an already typed configuration value."""
        if value is None and self.default is None and not self.required:
            return None
        try:
            if self.type is bool:
                if isinstance(value, str) and value.lower() in {"true", "false"}:
                    value = value.lower() == "true"
                if not isinstance(value, bool):
                    raise ValueError("expected true or false")
            elif self.type is int:
                if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value):
                    value = int(value)
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError("expected an integer")
            elif self.type is float:
                if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                    raise ValueError("expected a number")
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError("expected a finite number")
            elif self.type is str:
                if isinstance(value, PathLike):
                    value = str(value)
                if not isinstance(value, str):
                    raise ValueError("expected text")
            else:
                raise TypeError(f"Unsupported option type: {self.type}")
            if self.choices is not None and value not in self.choices:
                raise ValueError(f"expected one of {self.choices}")
            if self.minimum is not None and value < self.minimum:
                raise ValueError(f"must be at least {self.minimum}")
        except (ValueError, TypeError) as error:
            raise ValueError(f"{name}: {error}") from error
        return value


def resolve_options(
    definitions: Mapping[str, Option],
    values: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge defaults, file/API values, then explicit overrides; validate all keys."""
    merged = {}
    for source in (values, overrides):
        if source is None:
            continue
        if not isinstance(source, Mapping):
            raise ValueError("Configuration section must be a mapping")
        unknown = set(source) - set(definitions)
        if unknown:
            raise ValueError(f"Unknown configuration options: {', '.join(sorted(unknown))}")
        merged.update(source)
    resolved = {}
    for name, option in definitions.items():
        if name not in merged and option.required:
            raise ValueError(f"Missing required option: {name}")
        resolved[name] = option.convert(merged.get(name, option.default), name)
    return resolved


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a TOML run configuration; path resolution belongs to the runtime."""
    with Path(path).open("rb") as stream:
        try:
            return tomllib.load(stream)
        except tomllib.TOMLDecodeError as error:
            raise ValueError(f"{path}: {error}") from error


def load_legacy_parameters(path: str | Path) -> dict[str, int | float]:
    """Read legacy ``name = number`` parameters with line-specific errors.

    Native parameter names, types and ranges are validated by the solver.
    Blank lines and ``#`` comments are accepted; duplicate keys are rejected.
    """
    parameters = {}
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            if line.count("=") != 1:
                raise ValueError("expected name = number")
            name, raw = (part.strip() for part in line.split("=", 1))
            if not name or not raw:
                raise ValueError("expected name = number")
            if name in parameters:
                raise ValueError(f"duplicate parameter {name!r}")
            value = int(raw) if re.fullmatch(r"[+-]?\d+", raw) else float(raw)
            if not math.isfinite(value):
                raise ValueError("expected a finite number")
            parameters[name] = value
        except (ValueError, OverflowError) as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
    return parameters
