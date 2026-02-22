"""
Evaluation metrics for multi-class imbalanced classification.

Goes beyond simple accuracy to measure what matters scientifically:
  - Per-class recall: Did we find the rare supernovae?
  - Macro F1: Balanced performance across all classes
  - ROC-AUC: Discrimination ability independent of threshold
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_proba: Optional[List[List[float]]] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute a comprehensive set of classification metrics.

    Args:
        y_true:      Ground truth labels (integers)
        y_pred:      Predicted labels (integers)
        y_proba:     Predicted probabilities shape (N, num_classes) — for ROC-AUC
        class_names: Human-readable class names for per-class metrics

    Returns:
        Flat dict of metric_name → scalar value.
        Per-class metrics prefixed with class name, e.g. "galaxy_f1".

    Example:
        metrics = compute_metrics(y_true, y_pred, y_proba, class_names=["star", "galaxy"])
        print(metrics["macro_f1"])
        print(metrics["supernova_recall"])
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics: Dict[str, float] = {}

    # Overall accuracy
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    # Macro metrics (treat all classes equally — penalizes poor rare-class performance)
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["macro_precision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["macro_recall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # Weighted metrics (weighted by class frequency — dominated by majority class)
    metrics["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)

    num_classes = len(np.unique(y_true))
    names = class_names or [str(i) for i in range(num_classes)]

    for i, name in enumerate(names):
        if i < len(per_class_f1):
            metrics[f"{name}_f1"] = float(per_class_f1[i])
            metrics[f"{name}_precision"] = float(per_class_prec[i])
            metrics[f"{name}_recall"] = float(per_class_recall[i])

    # ROC-AUC (requires probability scores)
    if y_proba is not None:
        y_proba_np = np.array(y_proba)
        try:
            # Multi-class AUC: one-vs-rest, macro averaged
            metrics["roc_auc_macro"] = float(
                roc_auc_score(y_true, y_proba_np, multi_class="ovr", average="macro")
            )
            # Per-class AUC
            for i, name in enumerate(names):
                if i < y_proba_np.shape[1]:
                    binary_true = (y_true == i).astype(int)
                    if binary_true.sum() > 0 and (1 - binary_true).sum() > 0:
                        metrics[f"{name}_auc"] = float(
                            roc_auc_score(binary_true, y_proba_np[:, i])
                        )
        except ValueError as e:
            logger.warning(f"ROC-AUC computation failed: {e}")

    return metrics


def print_classification_report(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None,
) -> str:
    """Print and return sklearn's full classification report."""
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    logger.info(f"\n{report}")
    return report


def compute_confusion_matrix_normalized(
    y_true: List[int],
    y_pred: List[int],
    num_classes: int,
) -> np.ndarray:
    """
    Compute row-normalized confusion matrix.

    Row-normalized: entry [i, j] = fraction of class i predicted as class j.
    Diagonal = per-class recall. Makes imbalanced datasets interpretable.
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    # Normalize rows (true classes) — avoid division by zero
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = cm.astype(float) / np.maximum(row_sums, 1)
    return cm_normalized
