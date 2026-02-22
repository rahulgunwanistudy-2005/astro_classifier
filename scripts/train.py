#!/usr/bin/env python3
"""
train.py — Main training entrypoint.

Usage:
    # Train from scratch
    python scripts/train.py --config configs/base_config.yaml

    # Run experiment with overrides
    python scripts/train.py --config configs/experiment_focal.yaml

    # Resume from checkpoint
    python scripts/train.py --config configs/base_config.yaml \\
        --resume outputs/checkpoints/last.pth

    # Override config values on the fly (for quick experiments)
    python scripts/train.py --config configs/base_config.yaml \\
        --override training.lr=0.0001 loss.focal.gamma=3.0
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so imports work from anywhere
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataloader import compute_class_weights_tensor, create_dataloaders
from src.data.dataset import AstroDataset, build_train_transforms
from src.models.model_factory import build_model
from src.training.losses import build_criterion
from src.training.scheduler import build_optimizer, build_scheduler
from src.training.trainer import Trainer, set_seed
from src.utils.config import load_config, save_config
from src.utils.logger import get_logger, setup_root_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AstroClassifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Config overrides in dot.notation=value format, e.g. training.lr=0.001"
    )
    return parser.parse_args()


def apply_overrides(config, overrides: list) -> None:
    """Apply CLI key=value overrides to config in-place."""
    for override in overrides:
        if "=" not in override:
            logger.warning(f"Skipping malformed override: {override}")
            continue
        key_path, value = override.split("=", 1)
        keys = key_path.split(".")

        # Navigate to nested key
        obj = config
        for k in keys[:-1]:
            obj = obj[k]

        # Type inference: try int, float, bool, then string
        final_key = keys[-1]
        try:
            if value.lower() in ("true", "false"):
                obj[final_key] = value.lower() == "true"
            elif "." in value:
                obj[final_key] = float(value)
            else:
                obj[final_key] = int(value)
        except ValueError:
            obj[final_key] = value

        logger.info(f"Config override: {key_path} = {obj[final_key]}")


def main() -> None:
    args = parse_args()

    # Load and patch config 
    config = load_config(args.config)
    if args.override:
        apply_overrides(config, args.override)

    # Setup 
    setup_root_logger(log_file=f"outputs/logs/{config.project.experiment_name}.log")
    set_seed(config.project.seed)

    logger.info("=" * 70)
    logger.info(f"AstroClassifier Training: {config.project.experiment_name}")
    logger.info(f"Config: {args.config}")
    logger.info("=" * 70)

    # Save experiment config for reproducibility
    save_config(config, f"outputs/configs/{config.project.experiment_name}.yaml")

    # Data
    dataloaders = create_dataloaders(config)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # Compute class weights from training set for loss function
    train_dataset = train_loader.dataset
    class_weights = compute_class_weights_tensor(
        dataset=train_dataset,
        device=config.project.device if config.project.device != "cuda" else "cpu",
    )

    # Model 
    model = build_model(config)

    # Optimizer + Loss + Scheduler 
    optimizer = build_optimizer(model, config)
    criterion = build_criterion(config, class_weights=class_weights)

    # OneCycleLR needs total steps upfront
    num_training_steps = len(train_loader) * config.training.epochs
    scheduler = build_scheduler(
        optimizer=optimizer,
        config=config,
        num_training_steps=num_training_steps,
    )

    # Train 
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        config=config,
    )

    best_metrics = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        resume_from=args.resume,
    )

    logger.info("Training complete!")
    logger.info(f"Best macro F1: {best_metrics.get('macro_f1', 0):.4f}")
    logger.info(
        f"Best supernova recall: {best_metrics.get('supernova_recall', 0):.4f}"
    )
    logger.info(
        f"Checkpoint saved: {config.checkpointing.output_dir}/best_model.pth"
    )


if __name__ == "__main__":
    main()
