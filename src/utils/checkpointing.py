"""
Checkpointing: save/load model state with full reproducibility metadata.

Saves:
  - model state_dict
  - optimizer state_dict
  - scheduler state_dict
  - epoch, global_step
  - best metric value
  - full config (for audit trail)
  - random seeds

Maintains a rolling top-k checkpoints sorted by monitored metric.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CheckpointMeta:
    epoch: int
    global_step: int
    metric_name: str
    metric_value: float
    config_hash: str
    torch_version: str


def _hash_config(config: dict) -> str:
    """MD5 hash of the config dict for audit trail."""
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.md5(serialized.encode()).hexdigest()[:8]


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    global_step: int,
    metric_name: str,
    metric_value: float,
    output_dir: str | Path,
    config: dict,
    filename: Optional[str] = None,
    save_top_k: int = 3,
    mode: str = "max",
) -> Path:
    """
    Save a model checkpoint with full metadata.

    Args:
        model:        The PyTorch model to save
        optimizer:    Optimizer state (for resuming training)
        scheduler:    LR scheduler state
        epoch:        Current epoch number
        global_step:  Total training steps so far
        metric_name:  Name of the tracked metric (e.g. "val_macro_f1")
        metric_value: Current value of the metric
        output_dir:   Directory to save checkpoints
        config:       Full config dict (for reproducibility)
        filename:     Override filename (default: auto-generated)
        save_top_k:   Keep only the top-k checkpoints
        mode:         "max" or "min" — direction of metric improvement

    Returns:
        Path to the saved checkpoint file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = CheckpointMeta(
        epoch=epoch,
        global_step=global_step,
        metric_name=metric_name,
        metric_value=metric_value,
        config_hash=_hash_config(config),
        torch_version=torch.__version__,
    )

    checkpoint = {
        "meta": asdict(meta),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "config": config,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }

    if filename is None:
        filename = f"checkpoint_epoch{epoch:03d}_{metric_name}={metric_value:.4f}.pth"

    save_path = output_dir / filename
    torch.save(checkpoint, save_path)
    logger.info(f"Saved checkpoint: {save_path}")

    # Enforce top-k policy
    _prune_checkpoints(output_dir, save_top_k, mode)

    # Also save "last" checkpoint for easy resuming
    last_path = output_dir / "last.pth"
    torch.save(checkpoint, last_path)

    return save_path


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
    strict: bool = True,
) -> dict:
    """
    Load a checkpoint and restore model (and optionally optimizer/scheduler) state.

    Args:
        checkpoint_path: Path to the .pth file
        model:           Model to load weights into
        optimizer:       Optional optimizer to restore state
        scheduler:       Optional scheduler to restore state
        device:          Device to map tensors to
        strict:          Whether to strictly enforce state dict key matching

    Returns:
        The full checkpoint dict (includes meta, config, etc.)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    meta = checkpoint.get("meta", {})
    logger.info(
        f"Restored: epoch={meta.get('epoch')}, "
        f"{meta.get('metric_name')}={meta.get('metric_value', 0):.4f}"
    )

    return checkpoint


def _prune_checkpoints(output_dir: Path, keep_top_k: int, mode: str) -> None:
    """Remove checkpoints beyond top-k, ordered by metric value in filename."""
    ckpts = [
        f for f in output_dir.glob("checkpoint_epoch*.pth")
    ]
    if len(ckpts) <= keep_top_k:
        return

    def _extract_metric(path: Path) -> float:
        # Filename format: checkpoint_epoch005_val_macro_f1=0.8612.pth
        try:
            return float(path.stem.split("=")[-1])
        except (ValueError, IndexError):
            return 0.0

    reverse = mode == "max"
    ckpts_sorted = sorted(ckpts, key=_extract_metric, reverse=reverse)

    for old_ckpt in ckpts_sorted[keep_top_k:]:
        old_ckpt.unlink()
        logger.debug(f"Pruned checkpoint: {old_ckpt.name}")
