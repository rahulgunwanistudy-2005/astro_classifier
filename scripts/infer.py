#!/usr/bin/env python3
"""
infer.py — Classify a single astronomical image.

Usage:
    python scripts/infer.py \\
        --image path/to/galaxy.jpg \\
        --checkpoint outputs/checkpoints/best_model.pth \\
        --config configs/base_config.yaml

Output:
    Predicted class + confidence scores for all classes.
    Saves a visualization PNG if --output is specified.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from PIL import Image

from src.data.dataset import build_val_transforms
from src.models.model_factory import build_model
from src.utils.checkpointing import load_checkpoint
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify a single astronomical image")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--output", default=None, help="Optional: save prediction visualization to this path")
    return parser.parse_args()


def predict_single_image(
    image_path: str,
    model: torch.nn.Module,
    config,
    device: torch.device,
) -> dict:
    """
    Run inference on a single image.

    Returns:
        Dict with 'predicted_class', 'confidence', and 'class_probabilities'
    """
    transform = build_val_transforms(config)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)  # Add batch dim

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    class_names = config.data.classes
    pred_idx = probs.argmax().item()

    return {
        "predicted_class": class_names[pred_idx],
        "confidence": float(probs[pred_idx]),
        "class_probabilities": {
            cls: float(probs[i]) for i, cls in enumerate(class_names)
        },
    }


def visualize_prediction(
    image_path: str,
    result: dict,
    output_path: str,
) -> None:
    """Save a visualization of the prediction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(10, 4))

    # Original image
    img = Image.open(image_path).convert("RGB")
    ax_img.imshow(img)
    ax_img.set_title(
        f"Predicted: {result['predicted_class'].upper()}\n"
        f"Confidence: {result['confidence']:.1%}",
        fontweight="bold", fontsize=13,
    )
    ax_img.axis("off")

    # Probability bar chart
    classes = list(result["class_probabilities"].keys())
    probs = list(result["class_probabilities"].values())
    colors = ["#4CAF50" if c == result["predicted_class"] else "#90A4AE" for c in classes]

    bars = ax_bar.barh(classes, probs, color=colors, edgecolor="white", height=0.6)
    ax_bar.set_xlabel("Probability", fontweight="bold")
    ax_bar.set_title("Class Probabilities", fontweight="bold")
    ax_bar.set_xlim(0, 1)
    ax_bar.grid(axis="x", alpha=0.3)

    for bar, prob in zip(bars, probs):
        ax_bar.text(
            min(prob + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
            f"{prob:.3f}", va="center", fontsize=10,
        )

    plt.suptitle("AstroClassifier — Single Image Prediction", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    logger.info(f"Visualization saved: {output_path}")


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = build_model(config)
    load_checkpoint(args.checkpoint, model=model, device=str(device))
    model = model.to(device)

    # Infer
    result = predict_single_image(args.image, model, config, device)

    # Print results
    print("\n" + "=" * 50)
    print("ASTROCLASSIFIER PREDICTION")
    print("=" * 50)
    print(f"  Image:     {args.image}")
    print(f"  Predicted: {result['predicted_class'].upper()}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print()
    print("  Class probabilities:")
    for cls, prob in sorted(result["class_probabilities"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"    {cls:12s}: {prob:6.2%} {bar}")
    print("=" * 50)

    # Visualize
    if args.output:
        visualize_prediction(args.image, result, args.output)


if __name__ == "__main__":
    main()
