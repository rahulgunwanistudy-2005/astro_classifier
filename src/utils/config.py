"""
Config loader: reads YAML configs with base-config inheritance.

Usage:
    config = load_config("configs/experiment_focal.yaml")
    print(config.training.lr)   # dot-access on nested dicts
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


class DotDict(dict):
    """
    A dict subclass that supports dot-notation access for nested dicts.

    Example:
        cfg = DotDict({"model": {"lr": 0.001}})
        cfg.model.lr  # → 0.001
    """

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
            if isinstance(val, dict):
                return DotDict(val)
            return val
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge `override` into `base`.
    Values in `override` take precedence. Nested dicts are merged, not replaced.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key == "_base_":
            continue
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(config_path: str | Path) -> DotDict:
    """
    Load a YAML config file, resolving `_base_` inheritance.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        DotDict: Merged config with dot-access support.

    Example:
        config = load_config("configs/experiment_focal.yaml")
        print(config.training.batch_size)
    """
    config_path = Path(config_path).resolve()

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Config file is empty: {config_path}")

    # Resolve base config inheritance
    if "_base_" in raw:
        base_path = config_path.parent / raw["_base_"]
        base_config = load_config(base_path)  # Recursive — supports chained inheritance
        merged = _deep_merge(dict(base_config), raw)
    else:
        merged = raw

    return DotDict(merged)


def save_config(config: DotDict | dict, output_path: str | Path) -> None:
    """Save a config dict back to YAML (useful for logging experiment configs)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(dict(config), f, default_flow_style=False, sort_keys=False)


def config_to_flat_dict(config: DotDict | dict, prefix: str = "") -> dict[str, Any]:
    """
    Flatten nested config into a flat dict for W&B logging.

    Example:
        {"training": {"lr": 0.001}} → {"training.lr": 0.001}
    """
    flat = {}
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(config_to_flat_dict(value, prefix=full_key))
        else:
            flat[full_key] = value
    return flat


if __name__ == "__main__":
    # Quick smoke test
    import sys

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/base_config.yaml"
    cfg = load_config(cfg_path)
    print(f"Loaded config: {cfg_path}")
    print(f"  Project: {cfg.project.name}")
    print(f"  Model:   {cfg.model.name}")
    print(f"  Loss:    {cfg.loss.name}")
    print(f"  LR:      {cfg.optimizer.lr}")
    print(f"  Epochs:  {cfg.training.epochs}")
