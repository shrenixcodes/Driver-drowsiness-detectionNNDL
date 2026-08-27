"""
Evaluation pipeline: computes accuracy, precision, recall, F1, confusion
matrix and single-frame inference latency for each trained model on its held
-out test split, then writes a side-by-side comparison table.

Usage
-----
    python src/evaluate.py --model both

Outputs (under outputs/):
  figures/<name>_confusion_matrix.png
  metrics/<name>_classification_report.txt
  metrics/comparison_table.csv / .json   <- read directly by the Streamlit app

If a model has not been trained yet (no checkpoint file), it is skipped with
a clear warning instead of crashing, and the comparison table only contains
rows for models that were actually evaluated - no fabricated numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from src import config
from src.cnn_model import build_model_a
from src.data_loader import DatasetNotFoundError, build_frame_datasets, build_sequence_dataset
from src.lstm_model import build_model_b
from src.utils import checkpoint_exists, get_device, get_logger, load_checkpoint

logger = get_logger(__name__)


def _metrics_from_predictions(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def _save_confusion_matrix(y_true, y_pred, name: str) -> Path:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.2))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=config.DROWSY_LABEL_NAMES, yticklabels=config.DROWSY_LABEL_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{name} - Confusion Matrix")
    fig.tight_layout()
    out_path = config.FIGURES_DIR / f"{name}_confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _save_classification_report(y_true, y_pred, name: str) -> Path:
    report = classification_report(y_true, y_pred, target_names=config.DROWSY_LABEL_NAMES, zero_division=0)
    out_path = config.METRICS_DIR / f"{name}_classification_report.txt"
    out_path.write_text(report)
    logger.info(f"\n[{name}] Classification report:\n{report}")
    return out_path


@torch.no_grad()
def evaluate_model_a(device: torch.device) -> Optional[dict]:
    name = "CNN (Model A)"
    if not checkpoint_exists(config.CNN_CHECKPOINT_PATH):
        logger.warning(f"Skipping {name}: no checkpoint at {config.CNN_CHECKPOINT_PATH}. Train it first with 'python src/train.py --model cnn'.")
        return None

    ckpt = load_checkpoint(config.CNN_CHECKPOINT_PATH)
    model = build_model_a(pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    try:
        test_ds = build_frame_datasets()["test"]
    except DatasetNotFoundError as exc:
        logger.warning(f"Skipping {name} evaluation: {exc}")
        return None

    loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    all_true, all_pred = [], []
    for inputs, labels in loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_pred.extend(preds.tolist())
        all_true.extend(labels.numpy().tolist())

    # Single-frame latency benchmark (batch size 1, matches real-time usage)
    single = test_ds[0][0].unsqueeze(0).to(device)
    for _ in range(5):
        model(single)  # warm-up
    n_runs = 50
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model(single)
    latency_ms = (time.perf_counter() - t0) / n_runs * 1000.0

    metrics = _metrics_from_predictions(all_true, all_pred)
    metrics["inference_time_ms"] = latency_ms
    metrics["n_test_samples"] = len(all_true)

    _save_confusion_matrix(all_true, all_pred, "cnn_baseline")
    _save_classification_report(all_true, all_pred, "cnn_baseline")
    logger.info(f"[{name}] {metrics}")
    return metrics


@torch.no_grad()
def evaluate_model_b(device: torch.device, sequence_length: int = config.SEQUENCE_LENGTH) -> Optional[dict]:
    name = "CNN + LSTM/GRU (Model B)"
    if not checkpoint_exists(config.LSTM_CHECKPOINT_PATH):
        logger.warning(f"Skipping {name}: no checkpoint at {config.LSTM_CHECKPOINT_PATH}. Train it first with 'python src/train.py --model lstm'.")
        return None

    ckpt = load_checkpoint(config.LSTM_CHECKPOINT_PATH)
    rnn_type = ckpt.get("rnn_type", config.RNN_TYPE)
    model = build_model_b(pretrained=False, freeze_backbone=True, rnn_type=rnn_type)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    try:
        _train_ds, _val_ds, test_ds = build_sequence_dataset(seq_len=sequence_length)
    except DatasetNotFoundError as exc:
        logger.warning(f"Skipping {name} evaluation: {exc}")
        return None

    loader = DataLoader(test_ds, batch_size=config.SEQUENCE_BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    all_true, all_pred = [], []
    for sequences, labels in loader:
        sequences = sequences.to(device)
        logits = model(sequences)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_pred.extend(preds.tolist())
        all_true.extend(labels.numpy().tolist())

    # Real-time incremental latency: cost of processing ONE new frame, i.e.
    # extracting its feature + one RNN forward pass over the cached window -
    # this mirrors exactly what src/inference.py does per webcam frame, and
    # is the fair, apples-to-apples counterpart to Model A's per-frame cost.
    single_seq = test_ds[0][0].to(device)  # (T, C, H, W)
    single_frame = single_seq[-1:].to(device)  # (1, C, H, W)
    cached_features = model.extract_features(single_seq).unsqueeze(0)  # (1, T, feature_dim)

    for _ in range(5):
        model.extract_features(single_frame)
        model.forward_from_features(cached_features)
    n_runs = 50
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model.extract_features(single_frame)
        model.forward_from_features(cached_features)
    latency_ms = (time.perf_counter() - t0) / n_runs * 1000.0

    metrics = _metrics_from_predictions(all_true, all_pred)
    metrics["inference_time_ms"] = latency_ms
    metrics["n_test_samples"] = len(all_true)

    _save_confusion_matrix(all_true, all_pred, "cnn_lstm_proposed")
    _save_classification_report(all_true, all_pred, "cnn_lstm_proposed")
    logger.info(f"[{name}] {metrics}")
    return metrics


def build_comparison_table(results: dict) -> pd.DataFrame:
    rows = []
    display_names = {"cnn": "CNN (Baseline)", "lstm": "CNN + LSTM/GRU (Proposed)"}
    for key, label in display_names.items():
        m = results.get(key)
        if m is None:
            continue
        rows.append(
            {
                "Model": label,
                "Accuracy": round(m["accuracy"], 4),
                "Precision": round(m["precision"], 4),
                "Recall": round(m["recall"], 4),
                "F1 Score": round(m["f1_score"], 4),
                "Inference Time (ms)": round(m["inference_time_ms"], 3),
                "Test Samples": m["n_test_samples"],
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained drowsiness detection models.")
    parser.add_argument("--model", choices=["cnn", "lstm", "both"], default="both")
    parser.add_argument("--sequence-length", type=int, default=config.SEQUENCE_LENGTH)
    args = parser.parse_args()

    device = get_device(config.DEVICE)
    results = {}

    if args.model in ("cnn", "both"):
        results["cnn"] = evaluate_model_a(device)
    if args.model in ("lstm", "both"):
        results["lstm"] = evaluate_model_b(device, sequence_length=args.sequence_length)

    table = build_comparison_table(results)
    if table.empty:
        logger.warning(
            "No models could be evaluated - train at least one model first "
            "('python src/train.py --model cnn' or '--model lstm')."
        )
        return

    csv_path = config.METRICS_DIR / "comparison_table.csv"
    json_path = config.METRICS_DIR / "comparison_table.json"
    table.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(table.to_dict(orient="records"), indent=2))

    logger.info("\n" + table.to_string(index=False))
    logger.info(f"Saved comparison table -> {csv_path}, {json_path}")


if __name__ == "__main__":
    main()
