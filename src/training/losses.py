"""
Loss functions for imbalanced multi-class classification.

Implementations:
  FocalLoss       — Down-weights easy examples, focuses on hard/rare samples
  LabelSmoothingCE — Prevents overconfident predictions, improves calibration
  build_criterion  — Factory function from config

Mathematical background:

Standard Cross-Entropy:
    CE(p_t) = -log(p_t)

Focal Loss (Lin et al., 2017 — RetinaNet):
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    where:
        p_t = model's predicted probability for the correct class
        α_t = class balancing weight (inverse frequency)
        γ   = focusing parameter (γ=0 → standard CE; γ=2 → typical FL)

    The term (1 - p_t)^γ:
        - When p_t → 1 (easy, well-classified): weight ≈ 0 (down-weight heavily)
        - When p_t → 0 (hard, misclassified):   weight ≈ 1 (keep loss signal strong)

This is exactly what we need: force the network to pay attention to
rare quasars/supernovae that it tends to classify confidently as stars.
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss for imbalanced classification.

    Args:
        gamma:     Focusing parameter. Higher γ → more focus on hard examples.
                   Typical range: [0.5, 5.0]. γ=2 is the standard default.
        alpha:     Class weighting factor. Can be:
                   - None: no class weighting
                   - float: uniform weight for all classes
                   - List[float]: per-class weights (must sum to num_classes, ideally)
                   - Tensor: pre-computed per-class weight tensor
        reduction: "mean" | "sum" | "none"
        num_classes: Number of classes (needed when alpha is None, to auto-compute)

    References:
        Lin, T.Y. et al. (2017). Focal Loss for Dense Object Detection. ICCV.
        https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[float, List[float], torch.Tensor]] = None,
        reduction: str = "mean",
        num_classes: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes

        # Register alpha as buffer for proper device handling
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (list, tuple)):
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        elif isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha.float())
        else:
            # Scalar float: uniform weight
            self.register_buffer(
                "alpha",
                torch.full((num_classes,), alpha, dtype=torch.float32) if num_classes else None
            )

        logger.info(f"FocalLoss: gamma={gamma}, alpha={alpha}, reduction={reduction}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss.

        Args:
            logits:  Raw model outputs, shape (B, C) — NOT softmax probabilities
            targets: Ground truth class indices, shape (B,)

        Returns:
            Scalar loss value (or per-sample tensor if reduction="none")
        """
        # Compute standard log-softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=1)  # (B, C)
        probs = torch.exp(log_probs)               # (B, C)

        # Gather the predicted probability for the correct class: p_t
        log_pt = log_probs.gather(1, targets.view(-1, 1)).squeeze(1)  # (B,)
        pt = probs.gather(1, targets.view(-1, 1)).squeeze(1)           # (B,)

        # Focusing term: (1 - p_t)^γ
        focal_weight = (1.0 - pt) ** self.gamma  # (B,)

        # Class balancing weight: α_t
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]  # (B,)
            focal_weight = alpha_t * focal_weight

        # Focal loss per sample
        loss = -focal_weight * log_pt  # (B,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss  # "none"

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}, reduction={self.reduction}"


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-Entropy with label smoothing.

    Replaces hard one-hot targets with soft targets:
        y_smooth = (1 - ε) * y_hard + ε / K

    Benefits:
    - Prevents the model from becoming overconfident (max logit → ∞)
    - Acts as a mild regularizer
    - Can improve calibration (predicted probabilities match actual frequencies)

    Args:
        smoothing: Label smoothing factor ε ∈ [0, 1). 0.1 is typical.
        reduction: "mean" | "sum" | "none"
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()
        assert 0.0 <= smoothing < 1.0, "Smoothing must be in [0, 1)"
        self.smoothing = smoothing
        self.reduction = reduction
        logger.info(f"LabelSmoothingCrossEntropy: smoothing={smoothing}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)

        # NLL loss for the true class
        nll = -log_probs.gather(1, targets.view(-1, 1)).squeeze(1)

        # Uniform distribution loss (KL to uniform) for smoothing
        smooth_loss = -log_probs.mean(dim=1)

        # Mix: (1 - ε) * NLL + ε * smooth_loss
        loss = (1.0 - self.smoothing) * nll + self.smoothing * smooth_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_criterion(config, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """
    Build the loss function from config.

    Args:
        config:        Loaded DotDict config
        class_weights: Optional pre-computed class weight tensor.
                       If None, uses config.loss.focal.alpha (or auto-computes).

    Returns:
        nn.Module: The configured loss criterion

    Example:
        criterion = build_criterion(config, class_weights=weight_tensor)
        loss = criterion(logits, targets)
    """
    loss_name = config.loss.name

    if loss_name == "focal":
        focal_cfg = config.loss.focal

        # Determine alpha: config value > auto-computed class_weights > None
        alpha = focal_cfg.alpha
        if alpha is None and class_weights is not None:
            # Normalize class weights to [0, 1] range for use as alpha
            alpha = class_weights / class_weights.sum() * len(class_weights)
            logger.info("FocalLoss: using auto-computed class weights as alpha")

        criterion = FocalLoss(
            gamma=focal_cfg.gamma,
            alpha=alpha,
            reduction=focal_cfg.reduction,
            num_classes=config.model.num_classes,
        )

    elif loss_name == "cross_entropy":
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=0.0,
        )
        logger.info(
            f"CrossEntropyLoss with {'class weights' if class_weights is not None else 'no weights'}"
        )

    elif loss_name == "label_smoothing_ce":
        criterion = LabelSmoothingCrossEntropy(
            smoothing=config.loss.label_smoothing
        )

    else:
        raise ValueError(
            f"Unknown loss: '{loss_name}'. "
            f"Choose: 'focal', 'cross_entropy', 'label_smoothing_ce'"
        )

    return criterion
