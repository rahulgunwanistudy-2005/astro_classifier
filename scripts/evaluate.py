#!/usr/bin/env python3
"""
evaluate.py — Standalone evaluation on test set.

Usage:
    python scripts/evaluate.py \\
        --config configs/base_config.yaml \\
        --checkpoint outputs/checkpoints/best_model.pth \\
        --output-dir outputs/eval
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataloader import create_dataloaders
from src.evaluation.evaluator import Evaluator
from src.models.model_factory import build_model
from src.utils.checkpointing import load_checkpoint
from src.utils.config import load_config
from src.utils.logger import get_logger, setup_root_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AstroClassifier")
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--output-dir", default="outputs/eval", help="Where to save results")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Data split to evaluate")
    parser.add_argument("--wandb", action="store_true", help="Upload results to W&B")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    setup_root_logger()
    config = load_config(args.config)

    logger.info(f"Evaluating: {args.checkpoint}")
    logger.info(f"Split: {args.split}")

    # Data
    dataloaders = create_dataloaders(config)
    loader = dataloaders[args.split]

    # Model
    model = build_model(config)
    load_checkpoint(
        checkpoint_path=args.checkpoint,
        model=model,
        device=config.project.device,
    )

    # Evaluate
    evaluator = Evaluator(model, config)
    metrics = evaluator.evaluate(
        loader=loader,
        output_dir=args.output_dir,
        log_to_wandb=args.wandb,
    )

    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
