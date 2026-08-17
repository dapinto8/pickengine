"""Tunable model parameters, loaded from pickengine.toml.

The engine modules keep their named constants as code defaults; this Config
overrides them at the CLI/runner layer without editing code. `pickengine
tune` writes the chosen parameters here. Delete the file to fall back to the
code defaults. Unknown keys fail loudly — a typo must not silently leave a
parameter at its default.
"""

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

from pickengine.engine.elo import K_FACTOR
from pickengine.engine.pitching import ELO_PER_FIP
from pickengine.engine.probability import BLEND_WEIGHT_MODEL
from pickengine.engine.selection import MIN_EV

DEFAULT_CONFIG_PATH = Path("./pickengine.toml")


@dataclass(frozen=True)
class Config:
    elo_k: float = K_FACTOR
    elo_per_fip: float = ELO_PER_FIP
    blend_weight_model: float = BLEND_WEIGHT_MODEL
    min_ev: float = MIN_EV


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Config from the TOML file's [model] section; code defaults if absent."""
    path = Path(path)
    if not path.exists():
        return Config()
    model = tomllib.loads(path.read_text(encoding="utf-8")).get("model", {})
    known = {f.name for f in fields(Config)}
    unknown = set(model) - known
    if unknown:
        raise ValueError(f"unknown keys in {path} [model]: {sorted(unknown)}")
    return Config(**model)


def save_config(config: Config, path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
    """Write the config as pickengine.toml. Returns the path."""
    path = Path(path)
    path.write_text(
        "# pickengine tuned parameters — written by `pickengine tune`.\n"
        "# Loaded by pipeline CLI commands; delete to fall back to code defaults.\n"
        "\n"
        "[model]\n"
        f"elo_k = {config.elo_k}\n"
        f"elo_per_fip = {config.elo_per_fip}\n"
        f"blend_weight_model = {config.blend_weight_model}\n"
        f"min_ev = {config.min_ev}\n",
        encoding="utf-8",
    )
    return path
