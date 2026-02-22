"""
Model factory: centralized registry for model instantiation.

Adding a new model:
    1. Implement it in src/models/
    2. Register it in MODEL_REGISTRY below
    3. Reference it by key in configs/base_config.yaml → model.name

This avoids long if-elif chains and makes model selection fully config-driven.
"""

from __future__ import annotations

from typing import Dict, Type

import torch.nn as nn

from src.models.astro_cnn import AstroCNN, PretrainedBackbone
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Registry 
# Maps config string → model class constructor
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "AstroCNN": AstroCNN,
    "PretrainedBackbone": PretrainedBackbone,
}


def build_model(config) -> nn.Module:
    """
    Instantiate a model from config.

    Args:
        config: Loaded DotDict config (must have config.model.name)

    Returns:
        Initialized nn.Module

    Raises:
        KeyError: If model name is not in registry

    Example:
        model = build_model(config)
        model = model.to(device)
    """
    model_name = config.model.name

    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise KeyError(
            f"Unknown model '{model_name}'. "
            f"Available models: {available}. "
            f"Register new models in src/models/model_factory.py"
        )

    model_cls = MODEL_REGISTRY[model_name]

    # Build kwargs from config
    if model_name == "AstroCNN":
        model = model_cls(
            num_classes=config.model.num_classes,
            in_channels=config.data.num_channels,
            channels=list(config.model.channels),
            dropout_rate=config.model.dropout_rate,
            use_batch_norm=config.model.use_batch_norm,
        )
    elif model_name == "PretrainedBackbone":
        model = model_cls(
            backbone_name=config.model.pretrained_backbone,
            num_classes=config.model.num_classes,
            dropout_rate=config.model.dropout_rate,
        )
    else:
        # Generic fallback: try passing num_classes
        model = model_cls(num_classes=config.model.num_classes)

    logger.info(f"Built model: {model_name}")
    return model


def register_model(name: str, model_cls: Type[nn.Module]) -> None:
    """Register a custom model class at runtime."""
    MODEL_REGISTRY[name] = model_cls
    logger.info(f"Registered model: {name}")
