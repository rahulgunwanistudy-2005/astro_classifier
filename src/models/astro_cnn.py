"""
AstroCNN: Custom Convolutional Neural Network for celestial object classification.

Architecture designed for astronomical imagery:
  - Deep enough to capture multi-scale features (pixel→morphology→structure)
  - BatchNorm for training stability with small-ish astronomical datasets
  - Global Average Pooling instead of flatten → translation invariance
  - Dropout for regularization (prevents overfitting to majority class quirks)
  - Configurable channel depths via YAML

Design rationale:
  Stars:      Point-spread-function shape, bright core
  Galaxies:   Extended morphology, spiral arms, elliptical structure
  Quasars:    Point-like but with broad emission (spectral, but spatial too)
  Supernovae: Transient brightness in host galaxy context
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConvBlock(nn.Module):
    """
    Standard convolutional building block:
        Conv2d → BatchNorm2d → ReLU → MaxPool2d → Dropout2d

    Args:
        in_channels:  Number of input feature maps
        out_channels: Number of output feature maps
        kernel_size:  Convolution kernel size
        pool_size:    MaxPool kernel size (0 to skip pooling)
        dropout_rate: Spatial dropout probability
        use_bn:       Whether to apply BatchNorm
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout_rate: float = 0.0,
        use_bn: bool = True,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,  # Same padding preserves spatial dims
                bias=not use_bn,           # No bias needed with BatchNorm
            )
        ]

        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.ReLU(inplace=True))

        if pool_size > 0:
            layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=pool_size))

        if dropout_rate > 0:
            layers.append(nn.Dropout2d(p=dropout_rate))  # Spatial dropout for feature maps

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ClassifierHead(nn.Module):
    """
    Multi-layer classification head after feature extraction.

    Dense → BN → ReLU → Dropout → Dense → output logits
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        dropout_rate: float = 0.4,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class AstroCNN(nn.Module):
    """
    Custom CNN backbone for astronomical image classification.

    Feature extraction:
        ConvBlock(3 → ch[0])    — low-level: edges, intensity gradients
        ConvBlock(ch[0] → ch[1]) — mid-level: textures, local patterns
        ConvBlock(ch[1] → ch[2]) — high-level: morphological structure
        ConvBlock(ch[2] → ch[3]) — semantic: object-level representation

    Global Average Pooling collapses spatial dims → rotation/translation invariant
    Classifier Head maps features → class logits

    Args:
        num_classes:   Number of output classes
        in_channels:   Number of input image channels (default: 3)
        channels:      List of feature map sizes per ConvBlock
        dropout_rate:  Dropout probability for regularization
        use_batch_norm: Whether to use BatchNorm in ConvBlocks
    """

    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 3,
        channels: Optional[List[int]] = None,
        dropout_rate: float = 0.4,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()

        if channels is None:
            channels = [32, 64, 128, 256]

        assert len(channels) >= 2, "Need at least 2 channel sizes"

        # Build convolutional backbone
        conv_layers: List[nn.Module] = []
        current_channels = in_channels

        for i, out_ch in enumerate(channels):
            # Apply pooling to first N-1 blocks to progressively reduce spatial dims
            # Last block: no pooling (preserve spatial resolution for GAP)
            pool = 2 if i < len(channels) - 1 else 0
            # Increase dropout depth in later layers
            block_dropout = dropout_rate * (i / len(channels))

            conv_layers.append(
                ConvBlock(
                    in_channels=current_channels,
                    out_channels=out_ch,
                    pool_size=pool,
                    dropout_rate=block_dropout,
                    use_bn=use_batch_norm,
                )
            )
            current_channels = out_ch

        self.backbone = nn.Sequential(*conv_layers)

        # Global Average Pooling: (B, C, H, W) → (B, C)
        # Better than flatten: invariant to input spatial size, fewer parameters
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.classifier = ClassifierHead(
            in_features=channels[-1],
            hidden_dim=channels[-1] // 2,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )

        # Weight initialization
        self._initialize_weights()

        # Log model summary
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"AstroCNN initialized | "
            f"channels={channels} | "
            f"params={n_params:,} ({n_params/1e6:.2f}M)"
        )

    def _initialize_weights(self) -> None:
        """Kaiming (He) initialization for Conv layers, Xavier for Linear."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Logits tensor of shape (B, num_classes)
            (NOT softmax — use with CrossEntropyLoss or FocalLoss)
        """
        features = self.backbone(x)           # (B, channels[-1], H', W')
        pooled = self.gap(features)            # (B, channels[-1], 1, 1)
        flat = pooled.view(pooled.size(0), -1) # (B, channels[-1])
        logits = self.classifier(flat)         # (B, num_classes)
        return logits

    def get_feature_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns feature maps before GAP.
        Useful for visualization (class activation maps).

        Returns:
            (logits, feature_maps) — both tensors
        """
        features = self.backbone(x)
        pooled = self.gap(features)
        flat = pooled.view(pooled.size(0), -1)
        logits = self.classifier(flat)
        return logits, features


class PretrainedBackbone(nn.Module):
    """
    Transfer learning wrapper: ImageNet-pretrained backbone + custom head.

    Use this when your dataset is small (<50k images).
    AstroCNN is better when you have large datasets or domain-specific features.

    Supported backbones: "resnet18", "resnet50", "efficientnet_b0", "efficientnet_b3"
    """

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        dropout_rate: float = 0.4,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        backbone, feature_dim = self._build_backbone(backbone_name)
        self.backbone = backbone

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info(f"Backbone frozen: {backbone_name}")

        self.classifier = ClassifierHead(
            in_features=feature_dim,
            hidden_dim=256,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"PretrainedBackbone({backbone_name}) | trainable params: {n_trainable:,}")

    def _build_backbone(self, name: str) -> Tuple[nn.Module, int]:
        if name == "resnet18":
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            feature_dim = m.fc.in_features
            m.fc = nn.Identity()
            return m, feature_dim
        elif name == "resnet50":
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            feature_dim = m.fc.in_features
            m.fc = nn.Identity()
            return m, feature_dim
        elif name == "efficientnet_b0":
            m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            feature_dim = m.classifier[1].in_features
            m.classifier = nn.Identity()
            return m, feature_dim
        else:
            raise ValueError(f"Unknown backbone: {name}. Choose: resnet18, resnet50, efficientnet_b0")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)
