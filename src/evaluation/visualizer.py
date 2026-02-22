"""
Visualizer: publication-quality plots for model evaluation.

Generates:
  - Confusion matrix (normalized, heatmap)
  - Per-class ROC curves with AUC scores
  - Precision-Recall curves (better for imbalanced classes)
  - Training curves (loss, metrics vs epoch)
  - Sample prediction grids

All plots are saved to disk and optionally logged to W&B.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Publication-quality styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "figure.facecolor": "white",
})

# Consistent class colors (colorblind-friendly)
CLASS_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]


def plot_confusion_matrix(
    cm_normalized: np.ndarray,
    class_names: List[str],
    output_path: Optional[str | Path] = None,
    title: str = "Confusion Matrix (Row-Normalized)",
) -> plt.Figure:
    """
    Plot a normalized confusion matrix heatmap.

    Row-normalized means the diagonal shows per-class recall.
    Off-diagonal entries show what classes get confused with what.

    Args:
        cm_normalized: Row-normalized confusion matrix, shape (C, C)
        class_names:   List of class name strings
        output_path:   Optional save path (PNG)
        title:         Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    mask = np.eye(len(class_names), dtype=bool)

    # Draw off-diagonal first (light red)
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Reds",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0, vmax=1,
        ax=ax,
        mask=mask,
        cbar=False,
        annot_kws={"size": 11},
    )

    # Draw diagonal (green — correct predictions)
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0, vmax=1,
        ax=ax,
        mask=~mask,
        cbar=True,
        annot_kws={"size": 11, "weight": "bold"},
    )

    ax.set_title(title, fontweight="bold", pad=15)
    ax.set_ylabel("True Class", fontweight="bold")
    ax.set_xlabel("Predicted Class", fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        logger.info(f"Saved confusion matrix: {output_path}")

    return fig


def plot_roc_curves(
    y_true: List[int],
    y_proba: np.ndarray,
    class_names: List[str],
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot per-class ROC curves (one-vs-rest) with AUC scores.

    Args:
        y_true:      Ground truth labels
        y_proba:     Predicted probabilities, shape (N, num_classes)
        class_names: Class name list
        output_path: Save path
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.array(y_true)
    y_proba = np.array(y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (cls, color) in enumerate(zip(class_names, CLASS_COLORS)):
        binary_true = (y_true == i).astype(int)
        if binary_true.sum() == 0:
            continue

        fpr, tpr, _ = roc_curve(binary_true, y_proba[:, i])
        auc = roc_auc_score(binary_true, y_proba[:, i])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC={auc:.3f})")

    # Random classifier baseline
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6, label="Random")

    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.set_title("ROC Curves — One-vs-Rest", fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        logger.info(f"Saved ROC curves: {output_path}")

    return fig


def plot_precision_recall_curves(
    y_true: List[int],
    y_proba: np.ndarray,
    class_names: List[str],
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot Precision-Recall curves.

    PR curves are more informative than ROC for highly imbalanced classes.
    A good classifier pushes the PR curve toward the top-right corner.
    """
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (cls, color) in enumerate(zip(class_names, CLASS_COLORS)):
        binary_true = (y_true == i).astype(int)
        if binary_true.sum() == 0:
            continue

        precision, recall, _ = precision_recall_curve(binary_true, y_proba[:, i])
        # Area under PR curve
        ap = np.trapezoid(precision[::-1], recall[::-1])
        ax.plot(recall, precision, color=color, lw=2, label=f"{cls} (AP={ap:.3f})")

        # Mark the class baseline (random classifier AP)
        baseline = binary_true.mean()
        ax.axhline(y=baseline, color=color, lw=1, linestyle=":", alpha=0.5)

    ax.set_xlabel("Recall", fontweight="bold")
    ax.set_ylabel("Precision", fontweight="bold")
    ax.set_title("Precision-Recall Curves", fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        logger.info(f"Saved PR curves: {output_path}")

    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_macro_f1s: List[float],
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot training loss + validation loss + macro-F1 over epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = list(range(1, len(train_losses) + 1))

    # Loss curves
    ax1.plot(epochs, train_losses, "#2196F3", lw=2, label="Train Loss")
    ax1.plot(epochs, val_losses, "#F44336", lw=2, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss", fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Macro F1
    ax2.plot(epochs, val_macro_f1s, "#4CAF50", lw=2, label="Val Macro-F1")
    ax2.axhline(y=max(val_macro_f1s), color="#4CAF50", linestyle="--", alpha=0.5,
                label=f"Best: {max(val_macro_f1s):.4f}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro F1")
    ax2.set_title("Validation Macro-F1 Score", fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 1])

    plt.suptitle("Training Progress", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        logger.info(f"Saved training curves: {output_path}")

    return fig