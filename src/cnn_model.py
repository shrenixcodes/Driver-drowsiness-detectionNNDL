"""
Model A (baseline): frame-level CNN classifier.
Also defines `MobileNetFeatureExtractor`, the shared backbone reused by
Model B (src/lstm_model.py) so both models are trained on comparable features.

Why MobileNetV2?
  * Designed specifically for mobile/edge/CPU inference (depthwise-separable
    convolutions) -> fits the "real-time on a normal laptop" requirement far
    better than ResNet50/VGG-scale backbones.
  * ~3.5M parameters, pretrained on ImageNet -> transfer learning gives a
    strong starting point even with a modest eye-image dataset, avoiding the
    need to train a large CNN from scratch.
  * Well supported by torchvision with pretrained weights bundled.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


def _load_mobilenet_v2_backbone(pretrained: bool = True) -> nn.Module:
    """Load MobileNetV2's convolutional feature layers.

    Falls back to random initialization (with a warning) if pretrained
    weights cannot be downloaded (e.g. no internet access) - training will
    still work, just without the transfer-learning head start.
    """
    if pretrained:
        try:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
            backbone = models.mobilenet_v2(weights=weights)
            logger.info("Loaded MobileNetV2 with ImageNet-pretrained weights.")
            return backbone.features
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Could not download pretrained MobileNetV2 weights ({exc}). "
                "Falling back to random initialization - accuracy will be lower "
                "until trained for longer."
            )
    backbone = models.mobilenet_v2(weights=None)
    return backbone.features


class MobileNetFeatureExtractor(nn.Module):
    """MobileNetV2 conv trunk -> global average pool -> projection to a
    compact `config.FEATURE_DIM`-d feature vector.

    Used standalone (with a classifier head) as Model A, and per-frame inside
    Model B to turn each frame of a sequence into a feature vector for the
    RNN.
    """

    def __init__(self, feature_dim: int = config.FEATURE_DIM, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = _load_mobilenet_v2_backbone(pretrained=pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        backbone_out_channels = 1280  # MobileNetV2's last channel depth
        self.project = nn.Sequential(
            nn.Linear(backbone_out_channels, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 3, H, W) -> (N, feature_dim)"""
        features = self.backbone(x)
        pooled = self.pool(features).flatten(1)
        return self.project(pooled)


class DrowsinessCNN(nn.Module):
    """Model A: single-frame eye-state classifier (ALERT vs DROWSY).

    Outputs 2 logits (index 0 = open/alert, index 1 = closed/drowsy) trained
    with CrossEntropyLoss; `predict_proba` returns P(drowsy) directly for use
    by the real-time inference pipeline.
    """

    def __init__(self, feature_dim: int = config.FEATURE_DIM, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        self.feature_extractor = MobileNetFeatureExtractor(feature_dim, pretrained, freeze_backbone)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.classifier(features)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        return probs[:, 1]  # P(drowsy)


def build_model_a(pretrained: bool = True, freeze_backbone: bool = False) -> DrowsinessCNN:
    return DrowsinessCNN(feature_dim=config.FEATURE_DIM, pretrained=pretrained, freeze_backbone=freeze_backbone)
