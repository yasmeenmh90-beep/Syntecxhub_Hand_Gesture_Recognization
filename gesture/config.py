"""Load config.yaml (gesture -> action bindings per mode, plus tuning values)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import yaml

DEFAULT_CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
COMMANDS = {"toggle_dry_run", "screenshot", "open_spotify"}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Binding:
    gesture: str
    label: str
    keys: tuple[str, ...] = ()
    command: str | None = None
    repeat: bool = False

    @property
    def action(self) -> str:
        return self.command or "+".join(self.keys)


def _parse_binding(gesture: str, spec, where: str) -> Binding:
    if not isinstance(spec, dict) or "label" not in spec:
        raise ConfigError(f"{where}.{gesture}: needs a mapping with a 'label'")
    kinds = [k for k in ("key", "keys", "command") if k in spec]
    if len(kinds) != 1:
        raise ConfigError(f"{where}.{gesture}: give exactly one of key / keys / command")
    command = spec.get("command")
    if command is not None and command not in COMMANDS:
        raise ConfigError(f"{where}.{gesture}: unknown command {command!r} (use one of {sorted(COMMANDS)})")
    if "key" in spec:
        keys = (str(spec["key"]),)
    elif "keys" in spec:
        if not isinstance(spec["keys"], list) or not spec["keys"]:
            raise ConfigError(f"{where}.{gesture}: 'keys' must be a non-empty list")
        keys = tuple(str(k) for k in spec["keys"])
    else:
        keys = ()
    return Binding(gesture, str(spec["label"]), keys, command, bool(spec.get("repeat", False)))


def load_config(path: str | None = None) -> dict:
    """Read and validate a config file (default: GESTURE_CONFIG env var or ./config.yaml)."""
    path = path or os.environ.get("GESTURE_CONFIG") or DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        raise ConfigError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    modes = cfg.get("modes")
    if not isinstance(modes, dict) or not modes:
        raise ConfigError("config needs a non-empty 'modes' section")
    for name in modes:
        bindings_for(cfg, name)  # validates every binding up front
    if cfg.get("default_mode", next(iter(modes))) not in modes:
        raise ConfigError(f"default_mode {cfg['default_mode']!r} is not one of {list(modes)}")
    return cfg


@lru_cache(maxsize=None)
def default_config() -> dict:
    return load_config()


def bindings_for(cfg: dict, mode: str) -> dict[str, Binding]:
    """Global bindings merged with the mode's own (mode wins; `null` disables)."""
    if mode not in cfg["modes"]:
        raise ConfigError(f"unknown mode {mode!r}")
    own = cfg["modes"][mode] or {}
    merged = {}  # mode's own bindings first (that's the order the legend shows), then globals
    for gesture, spec in own.items():
        if spec is not None:
            merged[gesture] = _parse_binding(gesture, spec, f"modes.{mode}")
    for gesture, spec in (cfg.get("global") or {}).items():
        if gesture not in own and spec is not None:
            merged[gesture] = _parse_binding(gesture, spec, "global")
    return merged


def setting(cfg: dict, section: str, key: str, default):
    return (cfg.get(section) or {}).get(key, default)
