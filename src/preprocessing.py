"""
Image preprocessing and augmentation transforms.

Two sets of torchvision transforms are provided:
  * `get_train_transforms()`  - includes augmentation for training robustness
    (small rotations, brightness/contrast jitter, horizontal flip, blur) to
    approximate webcam variability (lighting, head tilt, focus).
  * `get_eval_transforms()`   - deterministic resize + normalize, used for
    validation, test, and real-time inference so results are reproducible.

`preprocess_frame_for_model` is the single entry point used by the real-time
inference pipeline to turn a raw BGR OpenCV crop into a normalized tensor
ready for the CNN, keeping training and inference preprocessing consistent.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torchvision import transforms

from src import config


def get_train_transforms(img_size: int = config.EYE_IMG_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )


def get_eval_transforms(img_size: int = config.EYE_IMG_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )


_EVAL_TRANSFORM = get_eval_transforms()


def preprocess_frame_for_model(bgr_crop: np.ndarray) -> torch.Tensor:
    """Convert a raw BGR crop (as produced by face_detection.crop_region) into
    a normalized CHW tensor ready to feed into the CNN. Used at inference time
    so preprocessing exactly matches `get_eval_transforms`.
    """
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    return _EVAL_TRANSFORM(rgb)


def preprocess_batch_for_model(bgr_crops: list) -> torch.Tensor:
    """Stack a list of BGR crops into a single batch tensor (N, C, H, W)."""
    tensors = [preprocess_frame_for_model(c) for c in bgr_crops]
    return torch.stack(tensors, dim=0)


def extract_eye_sequence_from_video(
    video_path: str,
    out_dir: str,
    every_n_frames: int = 2,
    max_frames: int = 300,
) -> int:
    """Extract an ordered sequence of eye-crop JPEGs from a video file.

    Used by the dataset-preparation step for Model B (CNN+LSTM): given a raw
    driving video labeled "drowsy" or "alert" (e.g. from NTHU-DDD or YawDD,
    which the user must obtain themselves - see README for access
    instructions), this walks the video, runs the face/eye detector on every
    `every_n_frames`-th frame, and writes the cropped, resized eye images to
    `out_dir` in temporal order (000000.jpg, 000001.jpg, ...).

    Returns the number of frames written. Skips frames where no face/eye is
    detected rather than failing the whole video.
    """
    import os

    from src.face_detection import FaceLandmarkDetector

    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    detector = FaceLandmarkDetector(backend="auto")
    frame_idx, written = 0, 0
    try:
        while written < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % every_n_frames == 0:
                result = detector.process(frame)
                if result.success and result.left_eye_crop is not None:
                    out_path = os.path.join(out_dir, f"{written:06d}.jpg")
                    cv2.imwrite(out_path, result.left_eye_crop)
                    written += 1
            frame_idx += 1
    finally:
        cap.release()
        detector.close()

    return written
