"""Configuration: every threshold the algorithm uses comes from here.

The YAML file ``defaults.yaml`` is the single source of truth for default values. A
:class:`Config` is an immutable, attribute-accessible view over the nested dictionary,
with dotted-path overrides (``Config.load(overrides={"scoring.fault.c": 0.2})``).

Profiles
--------
``prototype`` reproduces the behaviour of the reference prototype exactly.
``v3`` switches on every boolean under ``fixes`` (the proposed fixes for the known
weaknesses listed in section 10 of the brief). Individual fixes can be toggled with
overrides in either profile.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterator, Mapping
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

__all__ = ["Config", "load_config", "DEFAULTS_PATH"]

DEFAULTS_PATH = Path(str(resources.files("celldx.config").joinpath("defaults.yaml")))


class ConfigError(KeyError):
    """Raised when a configuration key is missing or an override names an unknown key."""


class Section(Mapping[str, Any]):
    """Read-only attribute view of a nested mapping."""

    __slots__ = ("_data", "_path")

    def __init__(self, data: Mapping[str, Any], path: str = "") -> None:
        object.__setattr__(self, "_data", dict(data))
        object.__setattr__(self, "_path", path)

    def __getattr__(self, name: str) -> Any:
        try:
            value = self._data[name]
        except KeyError as exc:
            raise ConfigError(f"config key not found: {self._path + name}") from exc
        if isinstance(value, Mapping):
            return Section(value, f"{self._path}{name}.")
        return value

    def __getitem__(self, name: str) -> Any:
        return self.__getattr__(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover - immutability guard
        raise AttributeError("Config is immutable; use Config.with_overrides")

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)


class Config(Section):
    """Top-level configuration object."""

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        overrides: Mapping[str, Any] | None = None,
        profile: str | None = None,
    ) -> Config:
        """Load defaults, merge an optional user YAML file, then apply dotted overrides.

        A user file may contain any subset of keys; unknown keys raise ``ConfigError`` so a
        typo never silently falls back to a default.
        """
        with open(DEFAULTS_PATH, encoding="utf-8") as fh:
            data: dict[str, Any] = yaml.safe_load(fh)
        if path is not None:
            with open(path, encoding="utf-8") as fh:
                user = yaml.safe_load(fh) or {}
            _merge_strict(data, user, "")
        if profile is not None:
            data["profile"] = profile
        if data.get("profile") == "v3":
            for key, value in list(data["fixes"].items()):
                if isinstance(value, bool):
                    data["fixes"][key] = True
        elif data.get("profile") not in ("prototype", "v3"):
            raise ConfigError(f"unknown profile {data.get('profile')!r}")
        for dotted, value in (overrides or {}).items():
            _set_dotted(data, dotted, value)
        return cls(data)

    def with_overrides(self, overrides: Mapping[str, Any]) -> Config:
        data = self.to_dict()
        for dotted, value in overrides.items():
            _set_dotted(data, dotted, value)
        return Config(data)

    def hash(self) -> str:
        """Short SHA-256 of the canonical JSON form, stored in every report."""
        blob = json.dumps(self._data, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def _merge_strict(base: dict[str, Any], new: Mapping[str, Any], prefix: str) -> None:
    for key, value in new.items():
        if key not in base:
            raise ConfigError(f"unknown config key: {prefix}{key}")
        if isinstance(value, Mapping) and isinstance(base[key], dict):
            _merge_strict(base[key], value, f"{prefix}{key}.")
        else:
            base[key] = value


def _set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node: Any = data
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            raise ConfigError(f"unknown config key: {dotted}")
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        raise ConfigError(f"unknown config key: {dotted}")
    node[parts[-1]] = value


def load_config(
    path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    profile: str | None = None,
) -> Config:
    """Convenience wrapper around :meth:`Config.load`."""
    return Config.load(path, overrides, profile)
