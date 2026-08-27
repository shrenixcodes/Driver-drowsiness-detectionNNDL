"""
Lightweight Grad-CAM for the Model A CNN baseline.

Highlights which pixels of the eye crop most influenced the "drowsy" score,
giving a simple, honest form of explainability without adding a second
heavyweight dependency. Not applied to Model B: Grad-CAM is defined for a
single spatial feature map, and attributing an RNN's temporal decision back
to pixels is a materially harder problem that's out of scope for this
project (see README -> Limitations).
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.cnn_model import DrowsinessCNN
from src.preprocessing import preprocess_frame_for_model
from src.utils import get_logger

logger = get_logger(__name__)


class GradCAM:
    """Grad-CAM against the last convolutional feature map of MobileNetV2.

    `model.feature_extractor.backbone` is the full `mobilenet_v2().features`
    stack; its output (before global average pooling) is exactly the spatial
    feature map Grad-CAM needs.
    """

    def __init__(self, model: DrowsinessCNN):
        self.model = model
        self.target_layer = model.feature_extractor.backbone
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self._activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = 1) -> np.ndarray:
        """Returns a (h, w) heatmap in [0, 1], h/w matching the backbone's
        final feature map spatial size (7x7 for a 224x224 input).
        """
        self.model.eval()
        # Grad-CAM needs gradients to flow back to the target layer's output
        # even if the backbone is frozen (requires_grad=False on its
        # weights), so we force the *input* to require grad - that alone is
        # enough to build a differentiable path through a frozen backbone.
        input_tensor = input_tensor.clone().requires_grad_(True)

        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        score = logits[:, class_idx].sum()
        score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks did not fire - check that the target layer is reachable in forward().")

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = F.relu((weights * self._activations).sum(dim=1, keepdim=True))  # (1, 1, h, w)
        cam = cam.squeeze().cpu().numpy()

        cam -= cam.min()
        max_val = cam.max()
        if max_val > 1e-8:
            cam /= max_val
        return cam


def overlay_heatmap_on_crop(bgr_crop: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Resizes `cam` to the crop's resolution and alpha-blends a JET colormap
    heatmap on top of the original BGR eye crop for display in the UI.
    """
    h, w = bgr_crop.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    return cv2.addWeighted(heatmap, alpha, bgr_crop, 1 - alpha, 0)


def generate_gradcam_overlay(model: DrowsinessCNN, bgr_crop: np.ndarray, class_idx: int = 1) -> np.ndarray:
    """One-call convenience: eye crop -> Grad-CAM overlay, same crop size."""
    gradcam = GradCAM(model)
    tensor = preprocess_frame_for_model(bgr_crop).unsqueeze(0)
    cam = gradcam.generate(tensor, class_idx=class_idx)
    return overlay_heatmap_on_crop(bgr_crop, cam)
