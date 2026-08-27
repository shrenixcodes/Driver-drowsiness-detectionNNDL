"""
Dataset loading for both models.

Model A (frame-level baseline) reads `data/{train,validation,test}/{open_eye,closed_eye}/*.jpg`.
Model B (CNN+LSTM) reads temporally-ordered sequences from `data/sequences/{drowsy,alert}/<clip>/*.jpg`.

Both loaders:
  * fail with a clear, actionable error message (not a stack trace) when the
    dataset is missing or empty, per the "handle missing datasets gracefully"
    requirement;
  * auto-derive validation/test splits from the training set when those
    folders are empty, so the pipeline still runs with a minimal dataset
    layout (just `data/train/<class>/*`);
  * expose a class-balancing helper (`get_weighted_sampler`) for when the
    two classes are not evenly represented, which is common in eye-state
    datasets.
"""
from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, WeightedRandomSampler

from src import config
from src.preprocessing import get_eval_transforms, get_train_transforms
from src.utils import get_logger

logger = get_logger(__name__)

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


class DatasetNotFoundError(RuntimeError):
    """Raised when a required dataset directory is missing or empty.

    Carries a human-readable, actionable message that calling code (train.py,
    the Streamlit app) can surface directly to the user instead of a raw
    traceback.
    """


# --------------------------------------------------------------------------
# Frame-level eye-state dataset (Model A + feature source for Model B)
# --------------------------------------------------------------------------
def _scan_class_folder(root: Path, class_name: str) -> List[Path]:
    folder = root / class_name
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)


def _scan_split(root: Path) -> Dict[str, List[Path]]:
    return {cls: _scan_class_folder(root, cls) for cls in config.EYE_STATE_CLASSES}


class EyeStateDataset(Dataset):
    """Frame-level open/closed eye classification dataset.

    `samples` is a list of (path, label) where label 1 = closed_eye (drowsy
    indicator), 0 = open_eye (alert indicator) - see config.DROWSY_LABEL_NAMES.
    """

    def __init__(self, samples: List[Tuple[Path, int]], transform=None):
        if len(samples) == 0:
            raise DatasetNotFoundError(
                "No images found for this dataset split. Populate "
                f"'{config.TRAIN_DIR}' with '{config.EYE_STATE_CLASSES[0]}' and "
                f"'{config.EYE_STATE_CLASSES[1]}' subfolders of eye images before "
                "training. See README.md -> Dataset Preparation."
            )
        self.samples = samples
        self.transform = transform or get_eval_transforms()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(np.array(image))
        return tensor, torch.tensor(label, dtype=torch.long)

    def class_counts(self) -> Dict[str, int]:
        counts = {"open_eye": 0, "closed_eye": 0}
        for _, label in self.samples:
            counts["closed_eye" if label == 1 else "open_eye"] += 1
        return counts


def _label_for_class(class_name: str) -> int:
    return 1 if class_name == "closed_eye" else 0


def build_frame_datasets(
    train_transform=None,
    eval_transform=None,
    seed: int = config.RANDOM_SEED,
) -> Dict[str, EyeStateDataset]:
    """Build train/val/test EyeStateDataset objects.

    If `data/train/` has images but `data/validation/` and/or `data/test/`
    are empty, a stratified split is carved out of the training set instead
    of failing, so the project works with the minimal `data/train/<class>/`
    layout described in the README.
    """
    train_by_class = _scan_split(config.TRAIN_DIR)
    val_by_class = _scan_split(config.VAL_DIR)
    test_by_class = _scan_split(config.TEST_DIR)

    total_train = sum(len(v) for v in train_by_class.values())
    if total_train == 0:
        raise DatasetNotFoundError(
            f"No training images found under '{config.TRAIN_DIR}'.\n"
            "Expected structure:\n"
            f"  {config.TRAIN_DIR}/open_eye/*.jpg\n"
            f"  {config.TRAIN_DIR}/closed_eye/*.jpg\n"
            "See README.md -> Dataset Preparation for how to obtain and place "
            "the MRL Eye Dataset (or an equivalent open/closed eye dataset)."
        )

    val_present = sum(len(v) for v in val_by_class.values()) > 0
    test_present = sum(len(v) for v in test_by_class.values()) > 0

    train_samples: List[Tuple[Path, int]] = []
    val_samples: List[Tuple[Path, int]] = []
    test_samples: List[Tuple[Path, int]] = []

    rng = random.Random(seed)

    for class_name, paths in train_by_class.items():
        label = _label_for_class(class_name)
        paths = list(paths)
        rng.shuffle(paths)

        remaining = paths
        if not test_present and len(remaining) > 2:
            remaining, held_out_test = train_test_split(
                remaining, test_size=config.TEST_SPLIT, random_state=seed
            )
            test_samples += [(p, label) for p in held_out_test]
        if not val_present and len(remaining) > 2:
            remaining, held_out_val = train_test_split(
                remaining, test_size=config.VAL_SPLIT, random_state=seed
            )
            val_samples += [(p, label) for p in held_out_val]

        train_samples += [(p, label) for p in remaining]

    if val_present:
        for class_name, paths in val_by_class.items():
            val_samples += [(p, _label_for_class(class_name)) for p in paths]
    if test_present:
        for class_name, paths in test_by_class.items():
            test_samples += [(p, _label_for_class(class_name)) for p in paths]

    if not val_samples:
        logger.warning("Validation split is empty even after auto-split; using a copy of train (not ideal).")
        val_samples = train_samples[: max(1, len(train_samples) // 10)]
    if not test_samples:
        logger.warning("Test split is empty even after auto-split; using a copy of validation.")
        test_samples = val_samples

    logger.info(
        f"Dataset splits -> train: {len(train_samples)}, val: {len(val_samples)}, test: {len(test_samples)} "
        f"(validation folder provided: {val_present}, test folder provided: {test_present})"
    )

    return {
        "train": EyeStateDataset(train_samples, transform=train_transform or get_train_transforms()),
        "val": EyeStateDataset(val_samples, transform=eval_transform or get_eval_transforms()),
        "test": EyeStateDataset(test_samples, transform=eval_transform or get_eval_transforms()),
    }


def get_weighted_sampler(dataset: EyeStateDataset) -> WeightedRandomSampler:
    """Inverse-frequency class-balancing sampler for imbalanced eye-state data."""
    labels = np.array([label for _, label in dataset.samples])
    class_sample_counts = np.bincount(labels, minlength=2).astype(np.float64)
    class_sample_counts[class_sample_counts == 0] = 1.0  # avoid div-by-zero
    weights_per_class = 1.0 / class_sample_counts
    sample_weights = weights_per_class[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


# --------------------------------------------------------------------------
# Sequence dataset (Model B: CNN + LSTM/GRU)
# --------------------------------------------------------------------------
_SUBJECT_ID_RE = re.compile(r"^(s\d+)", re.IGNORECASE)


class SequenceDataset(Dataset):
    """Yields (sequence_tensor[T, C, H, W], drowsy_label) pairs for Model B.

    Two source modes, resolved automatically by `build_sequence_dataset`:

    1. "video" mode (preferred, scientifically valid): reads real temporally
       ordered frames produced by `preprocessing.extract_eye_sequence_from_video`
       under `data/sequences/{drowsy,alert}/<clip_name>/*.jpg`. Requires a
       video dataset with genuine drowsy/alert sequences (e.g. NTHU-DDD,
       YawDD - see README for access instructions, as these are access-gated
       and cannot be auto-downloaded).

    2. "pseudo" mode (fallback demo path so the pipeline is always runnable):
       built from the frame-level eye-state dataset by grouping same-subject
       images (parsed from MRL-style filenames, e.g. `s0001_...png`) into
       fixed-length windows. This provides *no genuine temporal dynamics*
       (the frames are not consecutive video frames) - it exists purely so
       Model B's training/evaluation code path can be exercised end-to-end
       without a restricted-access video dataset. This limitation is logged
       loudly and documented in README.md; do not present pseudo-mode results
       as evidence of temporal modeling quality.
    """

    def __init__(self, sequences: List[Tuple[List[Path], int]], transform=None, mode: str = "unknown"):
        if len(sequences) == 0:
            raise DatasetNotFoundError(
                "No sequences available to build the CNN+LSTM dataset. Either "
                f"place labeled video clips under '{config.SEQUENCE_DIR}/drowsy' and "
                f"'{config.SEQUENCE_DIR}/alert' (see README -> Dataset Preparation), "
                "or ensure the frame-level eye dataset under "
                f"'{config.TRAIN_DIR}' has enough images to build pseudo-sequences."
            )
        self.sequences = sequences
        self.transform = transform or get_eval_transforms()
        self.mode = mode

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        paths, label = self.sequences[idx]
        frames = []
        for p in paths:
            image = Image.open(p).convert("RGB")
            frames.append(self.transform(np.array(image)))
        sequence = torch.stack(frames, dim=0)  # (T, C, H, W)
        return sequence, torch.tensor(label, dtype=torch.long)


def _windows(items: List, length: int, stride: int) -> List[List]:
    return [items[i : i + length] for i in range(0, len(items) - length + 1, stride)]


def _build_video_sequences(seq_len: int, stride: int) -> List[Tuple[List[Path], int]]:
    sequences = []
    for label_name, label in (("alert", 0), ("drowsy", 1)):
        class_dir = config.SEQUENCE_DIR / label_name
        if not class_dir.exists():
            continue
        for clip_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            frames = sorted(p for p in clip_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)
            for window in _windows(frames, seq_len, stride):
                sequences.append((window, label))
    return sequences


def _build_pseudo_sequences(seq_len: int, seed: int) -> List[Tuple[List[Path], int]]:
    """Fallback: group same-subject eye-state images into pseudo-sequences.

    See `SequenceDataset` docstring for the important caveat about this mode.
    """
    train_by_class = _scan_split(config.TRAIN_DIR)
    rng = random.Random(seed)
    sequences: List[Tuple[List[Path], int]] = []

    for class_name, paths in train_by_class.items():
        label = _label_for_class(class_name)
        by_subject: Dict[str, List[Path]] = {}
        for p in paths:
            match = _SUBJECT_ID_RE.match(p.stem)
            subject = match.group(1).lower() if match else "unknown"
            by_subject.setdefault(subject, []).append(p)

        for subject, subject_paths in by_subject.items():
            subject_paths = list(subject_paths)
            rng.shuffle(subject_paths)
            sequences += [(w, label) for w in _windows(subject_paths, seq_len, seq_len)]

    return sequences


def _safe_stratified_split(items: List, labels: List, test_size: float, seed: int):
    """`train_test_split` with stratification, falling back to a plain
    (non-stratified) split when a class has too few members to stratify
    (scikit-learn requires >= 2 samples per class per split). This matters
    most for small or highly imbalanced sequence datasets, e.g. a handful of
    pseudo-sequences for a rarely-seen subject.
    """
    stratify = labels if len(set(labels)) > 1 else None
    try:
        return train_test_split(items, labels, test_size=test_size, random_state=seed, stratify=stratify)
    except ValueError:
        return train_test_split(items, labels, test_size=test_size, random_state=seed, stratify=None)


def build_sequence_dataset(
    seq_len: int = config.SEQUENCE_LENGTH,
    stride: int = config.SEQUENCE_STRIDE,
    transform=None,
    seed: int = config.RANDOM_SEED,
) -> Tuple[SequenceDataset, SequenceDataset, SequenceDataset]:
    """Build train/val/test SequenceDataset objects, preferring real video
    sequences and transparently falling back to pseudo-sequences.
    """
    sequences = _build_video_sequences(seq_len, stride)
    mode = "video"

    if not sequences:
        logger.warning(
            "No real video sequences found under "
            f"'{config.SEQUENCE_DIR}/{{drowsy,alert}}/<clip>/'. Falling back to "
            "PSEUDO-SEQUENCES built from the frame-level eye dataset. These do "
            "NOT contain genuine temporal dynamics - they only let the CNN+LSTM "
            "code path run end-to-end for demonstration. For scientifically "
            "valid temporal results, obtain a video dataset such as NTHU-DDD or "
            "YawDD (see README -> Dataset Preparation) and re-run "
            "prepare_sequences before training Model B."
        )
        sequences = _build_pseudo_sequences(seq_len, seed)
        mode = "pseudo"

    if not sequences:
        raise DatasetNotFoundError(
            "Could not build any sequences for the CNN+LSTM model from either "
            f"'{config.SEQUENCE_DIR}' or '{config.TRAIN_DIR}'. See README.md -> "
            "Dataset Preparation."
        )

    labels = [label for _, label in sequences]
    relative_test = config.TEST_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
    train_seq, temp_seq, train_lbl, temp_lbl = _safe_stratified_split(
        sequences, labels, test_size=config.VAL_SPLIT + config.TEST_SPLIT, seed=seed,
    )
    val_seq, test_seq, _, _ = _safe_stratified_split(
        temp_seq, temp_lbl, test_size=relative_test, seed=seed,
    )

    logger.info(
        f"Sequence dataset ({mode} mode) -> train: {len(train_seq)}, val: {len(val_seq)}, "
        f"test: {len(test_seq)}, sequence_length={seq_len}"
    )

    return (
        SequenceDataset(train_seq, transform=transform or get_train_transforms(), mode=mode),
        SequenceDataset(val_seq, transform=transform or get_eval_transforms(), mode=mode),
        SequenceDataset(test_seq, transform=transform or get_eval_transforms(), mode=mode),
    )
