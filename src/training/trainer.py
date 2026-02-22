"""
Trainer: orchestrates the full training loop.

Features:
  - Automatic Mixed Precision (AMP) via torch.cuda.amp
  - Gradient clipping
  - Gradient accumulation
  - W&B logging (loss curves, metrics, sample predictions)
  - Best model checkpointing
  - Reproducible seeding
  - Early stopping (optional)
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics
from src.utils.checkpointing import save_checkpoint
from src.utils.config import config_to_flat_dict
from src.utils.logger import get_logger

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    """Ensure full reproducibility across all RNG sources."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # These two lines trade speed for determinism — comment out for faster training
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    """
    Full training loop with evaluation, logging, and checkpointing.

    Usage:
        trainer = Trainer(model, optimizer, scheduler, criterion, config)
        trainer.fit(train_loader, val_loader)

    Args:
        model:      Initialized PyTorch model
        optimizer:  Configured optimizer
        scheduler:  LR scheduler (or None)
        criterion:  Loss function
        config:     Loaded DotDict config
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        criterion: nn.Module,
        config,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.config = config

        # Device setup
        self.device = self._setup_device()
        self.model = self.model.to(self.device)

        # AMP scaler for mixed precision training
        self.scaler = GradScaler(enabled=config.training.mixed_precision and self.device.type == "cuda")

        # Training state
        self.global_step = 0
        self.best_metric = float("-inf") if config.checkpointing.mode == "max" else float("inf")
        self.start_epoch = 0

        # W&B setup
        self.use_wandb = config.logging.use_wandb
        self.wandb_run = None
        if self.use_wandb:
            self._init_wandb()

        logger.info(f"Trainer initialized | device={self.device}")

    def _setup_device(self) -> torch.device:
        requested = self.config.project.device
        if requested == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU: {gpu_name} ({vram:.1f} GB VRAM)")
        elif requested == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            if requested in ("cuda", "mps"):
                logger.warning(f"{requested.upper()} not available, falling back to CPU")
        return device

    def _init_wandb(self) -> None:
        try:
            import wandb

            self.wandb_run = wandb.init(
                project=self.config.logging.wandb_project,
                entity=self.config.logging.wandb_entity or None,
                name=self.config.project.experiment_name,
                config=config_to_flat_dict(self.config),
                resume="allow",
            )
            logger.info(f"W&B run: {self.wandb_run.url}")
        except ImportError:
            logger.warning("wandb not installed. Run: pip install wandb")
            self.use_wandb = False
        except Exception as e:
            logger.warning(f"W&B init failed: {e} — continuing without tracking")
            self.use_wandb = False

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_from: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Run the full training loop.

        Args:
            train_loader:  DataLoader for training data
            val_loader:    DataLoader for validation data
            resume_from:   Optional path to checkpoint to resume from

        Returns:
            Dict with best validation metrics
        """
        if resume_from:
            self._resume(resume_from)

        classes = self.config.data.classes
        epochs = self.config.training.epochs
        best_metrics = {}

        logger.info(f"Starting training: {self.start_epoch + 1} → {epochs} epochs")

        for epoch in range(self.start_epoch, epochs):
            epoch_start = time.time()

            # Training phase
            train_metrics = self._train_epoch(train_loader, epoch)

            # Validation phase 
            val_metrics = self._val_epoch(val_loader, epoch, classes)

            # LR scheduling 
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                if hasattr(self.scheduler, "step"):
                    from torch.optim.lr_scheduler import ReduceLROnPlateau
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics[self.config.checkpointing.monitor])
                    else:
                        self.scheduler.step()

            epoch_time = time.time() - epoch_start

            # Logging
            log_dict = {
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()},
                "lr": current_lr,
                "epoch": epoch,
                "epoch_time_s": epoch_time,
            }

            logger.info(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_macro_f1={val_metrics.get('macro_f1', 0):.4f} | "
                f"lr={current_lr:.2e} | "
                f"{epoch_time:.1f}s"
            )

            if self.use_wandb and self.wandb_run:
                self.wandb_run.log(log_dict, step=epoch)

            # Checkpointing
            monitor = self.config.checkpointing.monitor.replace("val_", "")
            current_metric = val_metrics.get(monitor, val_metrics.get("macro_f1", 0))
            is_best = self._is_better(current_metric)

            if is_best:
                self.best_metric = current_metric
                best_metrics = val_metrics.copy()
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch + 1,
                    global_step=self.global_step,
                    metric_name=self.config.checkpointing.monitor,
                    metric_value=current_metric,
                    output_dir=self.config.checkpointing.output_dir,
                    config=dict(self.config),
                    filename="best_model.pth",
                    save_top_k=self.config.checkpointing.save_top_k,
                    mode=self.config.checkpointing.mode,
                )
                logger.info(f"  ✓ New best: {self.config.checkpointing.monitor}={current_metric:.4f}")

            # Always save rolling checkpoint
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch + 1,
                global_step=self.global_step,
                metric_name=self.config.checkpointing.monitor,
                metric_value=current_metric,
                output_dir=self.config.checkpointing.output_dir,
                config=dict(self.config),
                save_top_k=self.config.checkpointing.save_top_k,
                mode=self.config.checkpointing.mode,
            )

        if self.use_wandb and self.wandb_run:
            self.wandb_run.finish()

        logger.info(f"Training complete. Best {self.config.checkpointing.monitor}: {self.best_metric:.4f}")
        return best_metrics

    def _train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        log_interval = self.config.logging.log_interval
        accumulation_steps = self.config.training.accumulation_steps
        clip_norm = self.config.training.clip_grad_norm

        pbar = tqdm(loader, desc=f"Train E{epoch+1}", leave=False, ncols=100)
        self.optimizer.zero_grad()

        for step, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Forward pass with AMP
            with autocast(enabled=self.config.training.mixed_precision and self.device.type == "cuda"):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss = loss / accumulation_steps  # Scale for gradient accumulation

            # Backward pass
            self.scaler.scale(loss).backward()

            # Gradient accumulation: only update every N steps
            if (step + 1) % accumulation_steps == 0:
                if clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # Step OneCycleLR per batch (not per epoch)
                if self.scheduler is not None:
                    from torch.optim.lr_scheduler import OneCycleLR
                    if isinstance(self.scheduler, OneCycleLR):
                        self.scheduler.step()

            # Track metrics
            actual_loss = loss.item() * accumulation_steps
            total_loss += actual_loss
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            self.global_step += 1

            # Batch-level logging
            if self.global_step % log_interval == 0:
                batch_acc = correct / max(total, 1)
                pbar.set_postfix(loss=f"{actual_loss:.4f}", acc=f"{batch_acc:.3f}")

        return {
            "loss": total_loss / len(loader),
            "accuracy": correct / max(total, 1),
        }

    @torch.no_grad()
    def _val_epoch(
        self, loader: DataLoader, epoch: int, classes: list
    ) -> Dict[str, float]:
        """Run one validation epoch and compute full metrics."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        for images, labels in tqdm(loader, desc=f"Val   E{epoch+1}", leave=False, ncols=100):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(enabled=self.config.training.mixed_precision and self.device.type == "cuda"):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        metrics = compute_metrics(
            y_true=all_labels,
            y_pred=all_preds,
            y_proba=all_probs,
            class_names=classes,
        )
        metrics["loss"] = total_loss / len(loader)

        return metrics

    def _is_better(self, current: float) -> bool:
        mode = self.config.checkpointing.mode
        if mode == "max":
            return current > self.best_metric
        return current < self.best_metric

    def _resume(self, checkpoint_path: str) -> None:
        from src.utils.checkpointing import load_checkpoint
        ckpt = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=str(self.device),
        )
        self.start_epoch = ckpt["meta"]["epoch"]
        self.global_step = ckpt["meta"]["global_step"]
        logger.info(f"Resumed from epoch {self.start_epoch}")
