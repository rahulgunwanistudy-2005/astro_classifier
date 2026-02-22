"""
Learning rate scheduler factory.

Strategies:
  cosine_warmup   — Linear warmup → cosine decay (default, state-of-the-art)
  one_cycle       — OneCycleLR: fast warmup, aggressive decay (good for sweeps)
  step            — StepLR: old-school, predictable
  reduce_on_plateau — ReduceLROnPlateau: adaptive, metric-driven

cosine_warmup is generally the best default for deep learning:
  - Linear warmup prevents gradient instability at the start
  - Cosine decay smoothly anneals LR without sharp drops
  - Reaches min_lr at end of training (not abruptly)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    OneCycleLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
    _LRScheduler,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_optimizer(model: nn.Module, config) -> Optimizer:
    """
    Build optimizer from config.

    Uses weight decay only on non-bias, non-BatchNorm parameters
    (standard best practice to avoid over-regularizing BN/bias terms).

    Args:
        model:  PyTorch model
        config: DotDict config

    Returns:
        Configured optimizer
    """
    opt_cfg = config.optimizer

    # Separate parameters: decay vs no-decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": opt_cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    name = opt_cfg.name.lower()
    if name == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=opt_cfg.lr,
            betas=tuple(opt_cfg.betas),
        )
    elif name == "adam":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=opt_cfg.lr,
            betas=tuple(opt_cfg.betas),
        )
    elif name == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=opt_cfg.lr,
            momentum=opt_cfg.momentum,
            nesterov=opt_cfg.nesterov,
        )
    else:
        raise ValueError(f"Unknown optimizer: '{name}'. Choose: adamw, adam, sgd")

    logger.info(
        f"Optimizer: {name} | lr={opt_cfg.lr} | "
        f"wd={opt_cfg.weight_decay} | "
        f"decay_params={len(decay_params)} | no_decay={len(no_decay_params)}"
    )
    return optimizer


def build_scheduler(
    optimizer: Optimizer,
    config,
    num_training_steps: Optional[int] = None,
) -> Optional[object]:
    """
    Build learning rate scheduler from config.

    Args:
        optimizer:           Configured optimizer
        config:              DotDict config
        num_training_steps:  Total training steps (needed for OneCycleLR)

    Returns:
        Scheduler instance, or None if scheduler is disabled
    """
    sched_cfg = config.scheduler
    name = sched_cfg.name.lower()
    total_epochs = config.training.epochs

    if name == "cosine_warmup":
        scheduler = _build_cosine_warmup(
            optimizer=optimizer,
            warmup_epochs=sched_cfg.warmup_epochs,
            total_epochs=total_epochs,
            min_lr=sched_cfg.min_lr,
            base_lr=config.optimizer.lr,
        )

    elif name == "one_cycle":
        if num_training_steps is None:
            raise ValueError("one_cycle scheduler requires num_training_steps")
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.optimizer.lr,
            total_steps=num_training_steps,
            pct_start=0.3,        # Spend 30% of training in warmup phase
            anneal_strategy="cos",
            div_factor=25.0,      # Initial LR = max_lr / 25
            final_div_factor=1e4, # Final LR = initial_lr / 10000
        )
        logger.info(f"OneCycleLR: max_lr={config.optimizer.lr}, steps={num_training_steps}")

    elif name == "step":
        scheduler = StepLR(
            optimizer,
            step_size=sched_cfg.step_size,
            gamma=sched_cfg.gamma,
        )
        logger.info(f"StepLR: step_size={sched_cfg.step_size}, gamma={sched_cfg.gamma}")

    elif name == "reduce_on_plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=config.checkpointing.mode,
            patience=sched_cfg.patience,
            factor=sched_cfg.factor,
            min_lr=sched_cfg.min_lr,
            verbose=True,
        )
        logger.info(
            f"ReduceLROnPlateau: patience={sched_cfg.patience}, factor={sched_cfg.factor}"
        )

    elif name == "none":
        return None

    else:
        raise ValueError(
            f"Unknown scheduler: '{name}'. "
            f"Choose: cosine_warmup, one_cycle, step, reduce_on_plateau, none"
        )

    return scheduler


def _build_cosine_warmup(
    optimizer: Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr: float,
    base_lr: float,
) -> SequentialLR:
    """
    Linear warmup from near-zero to base_lr, then cosine decay to min_lr.

    Why warmup?
    At initialization, gradients can be large and noisy. Starting with a
    small LR lets the model "settle" before full-speed training begins.
    """
    # Phase 1: Linear warmup (epochs 0 → warmup_epochs)
    warmup = LinearLR(
        optimizer,
        start_factor=1e-4,   # Start at LR = base_lr * 1e-4
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    # Phase 2: Cosine annealing (epochs warmup_epochs → total_epochs)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=min_lr,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )

    logger.info(
        f"CosineWarmup: {warmup_epochs} warmup epochs → "
        f"cosine decay to {min_lr:.2e} over {total_epochs - warmup_epochs} epochs"
    )
    return scheduler
