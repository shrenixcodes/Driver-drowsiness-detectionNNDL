"""
Model B (proposed): CNN feature extractor + LSTM/GRU temporal model.

Pipeline: sequence of eye crops -> per-frame MobileNetV2 feature vector ->
GRU (default) or LSTM over the `SEQUENCE_LENGTH`-frame window -> dense head
-> P(drowsy).

Why GRU by default (LSTM is a one-line config swap, `RNN_TYPE="LSTM"`)?
  * GRU has ~25% fewer parameters than an LSTM of the same hidden size (no
    separate cell state, one fewer gate) -> lower per-frame inference
    latency, which matters directly for the real-time-on-CPU requirement.
  * At the short sequence lengths used here (a few seconds of video), GRU and
    LSTM perform comparably in practice, so the latency advantage dominates
    the choice.

The model exposes `extract_features` and `forward_from_features` separately
(in addition to the standard `forward`) so the real-time inference loop can
cache each frame's CNN feature vector as it arrives and only run the cheap
RNN forward pass over the rolling window every frame, instead of re-running
the CNN backbone over the whole window each time.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src import config
from src.cnn_model import MobileNetFeatureExtractor
from src.utils import get_logger

logger = get_logger(__name__)


class CNNRNNDrowsinessModel(nn.Module):
    def __init__(
        self,
        feature_dim: int = config.FEATURE_DIM,
        rnn_type: str = config.RNN_TYPE,
        hidden_size: int = config.RNN_HIDDEN_SIZE,
        num_layers: int = config.RNN_NUM_LAYERS,
        dropout: float = config.RNN_DROPOUT,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.feature_extractor = MobileNetFeatureExtractor(feature_dim, pretrained, freeze_backbone)

        self.rnn_type = rnn_type.upper()
        rnn_cls = nn.LSTM if self.rnn_type == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
        )

    # ------------------------------------------------------------------
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C, H, W) single frame per sample -> (N, feature_dim).

        Exposed separately so real-time inference can compute one frame's
        feature and append it to a rolling buffer instead of recomputing the
        whole window every step.
        """
        return self.feature_extractor(x)

    def forward_from_features(self, feature_seq: torch.Tensor) -> torch.Tensor:
        """feature_seq: (N, T, feature_dim) -> (N, 2) logits."""
        rnn_out, _ = self.rnn(feature_seq)
        last_step = rnn_out[:, -1, :]
        return self.classifier(last_step)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, T, C, H, W) -> (N, 2) logits."""
        n, t, c, h, w = x.shape
        x = x.view(n * t, c, h, w)
        feats = self.feature_extractor(x)
        feats = feats.view(n, t, -1)
        return self.forward_from_features(feats)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.softmax(self.forward(x), dim=1)[:, 1]

    @torch.no_grad()
    def predict_proba_from_features(self, feature_seq: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.softmax(self.forward_from_features(feature_seq), dim=1)[:, 1]


def build_model_b(
    pretrained: bool = True,
    freeze_backbone: bool = True,
    rnn_type: str = config.RNN_TYPE,
) -> CNNRNNDrowsinessModel:
    return CNNRNNDrowsinessModel(
        feature_dim=config.FEATURE_DIM,
        rnn_type=rnn_type,
        hidden_size=config.RNN_HIDDEN_SIZE,
        num_layers=config.RNN_NUM_LAYERS,
        dropout=config.RNN_DROPOUT,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )
