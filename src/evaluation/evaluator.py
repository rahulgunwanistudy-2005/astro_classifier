"""
Evaluator: runs the complete evaluation suite on a trained model.

Produces:
  - Confusion matrix PNG
  - ROC curve PNG
  - Precision-Recall curve PNG
  - Full classification report (text)
  - Metrics JSON (for CI/CD comparison)
  - Optional W&B artifact upload
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import (
    compute_confusion_matrix_normalized,
    compute_metrics,
    print_classification_report,
)
from src.evaluation.visualizer import (
    plot_confusion_matrix,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_training_curves,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """
    End-to-end evaluation pipeline.

    Usage:
        evaluator = Evaluator(model, config)
        results = evaluator.evaluate(test_loader, output_dir="outputs/eval")
    """

    def __init__(self, model: nn.Module, config) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.classes = config.data.classes

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        output_dir: str | Path = "outputs/eval",
        log_to_wandb: bool = False,
    ) -> Dict[str, float]:
        """
        Run full evaluation and generate all artifacts.

        Args:
            loader:       DataLoader for evaluation data
            output_dir:   Directory to save plots and reports
            log_to_wandb: Whether to upload artifacts to W&B

        Returns:
            Dict of metric_name → float
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Running evaluation...")
        y_true, y_pred, y_proba = self._run_inference(loader)

        # Metrics
        metrics = compute_metrics(y_true, y_pred, y_proba, class_names=self.classes)

        # Classification report (printed + saved)
        report = print_classification_report(y_true, y_pred, class_names=self.classes)
        (output_dir / "classification_report.txt").write_text(report)

        # Confusion matrix
        cm = compute_confusion_matrix_normalized(
            y_true, y_pred, num_classes=len(self.classes)
        )
        plot_confusion_matrix(
            cm_normalized=cm,
            class_names=self.classes,
            output_path=output_dir / "confusion_matrix.png",
        )

        # ROC curves
        if y_proba is not None:
            y_proba_np = np.array(y_proba)
            plot_roc_curves(
                y_true=y_true,
                y_proba=y_proba_np,
                class_names=self.classes,
                output_path=output_dir / "roc_curves.png",
            )
            plot_precision_recall_curves(
                y_true=y_true,
                y_proba=y_proba_np,
                class_names=self.classes,
                output_path=output_dir / "pr_curves.png",
            )

        # Save metrics JSON
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics: {metrics_path}")

        # Pretty print key metrics
        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"  Accuracy:     {metrics['accuracy']:.4f}")
        logger.info(f"  Macro F1:     {metrics['macro_f1']:.4f}")
        logger.info(f"  Macro AUC:    {metrics.get('roc_auc_macro', 0):.4f}")
        logger.info("-" * 60)
        for cls in self.classes:
            logger.info(
                f"  {cls:12s}: P={metrics.get(f'{cls}_precision', 0):.3f}  "
                f"R={metrics.get(f'{cls}_recall', 0):.3f}  "
                f"F1={metrics.get(f'{cls}_f1', 0):.3f}  "
                f"AUC={metrics.get(f'{cls}_auc', 0):.3f}"
            )
        logger.info("=" * 60)

        # W&B artifact upload
        if log_to_wandb and self.config.logging.use_wandb:
            self._log_to_wandb(metrics, output_dir)

        return metrics

    def _run_inference(
        self, loader: DataLoader
    ) -> Tuple[List[int], List[int], List[List[float]]]:
        """Run model inference on all batches."""
        self.model.eval()
        all_labels, all_preds, all_probs = [], [], []

        for batch in tqdm(loader, desc="Evaluating", ncols=100):
            # Handle (images, labels) or (images, labels, paths) batches
            images, labels = batch[0], batch[1]
            images = images.to(self.device, non_blocking=True)

            with autocast(enabled=self.config.training.mixed_precision and self.device.type == "cuda"):
                logits = self.model(images)

            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

        return all_labels, all_preds, all_probs

    def _log_to_wandb(self, metrics: Dict[str, float], output_dir: Path) -> None:
        """Upload evaluation artifacts to W&B."""
        try:
            import wandb

            run = wandb.init(
                project=self.config.logging.wandb_project,
                name=f"{self.config.project.experiment_name}_eval",
                resume="allow",
                job_type="evaluation",
            )

            run.log(metrics)

            artifact = wandb.Artifact("evaluation_plots", type="results")
            for png in output_dir.glob("*.png"):
                artifact.add_file(str(png))
            run.log_artifact(artifact)

            logger.info(f"Evaluation artifacts uploaded to W&B: {run.url}")
            run.finish()

        except Exception as e:
            logger.warning(f"W&B upload failed: {e}")
