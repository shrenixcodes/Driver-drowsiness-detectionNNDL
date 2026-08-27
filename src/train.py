"""
Unified training pipeline for both models.

Usage
-----
    python src/train.py --model cnn                 # Model A only
    python src/train.py --model lstm                 # Model B only
    python src/train.py --model both                 # both, sequentially
    python src/train.py --model cnn --epochs 15 --batch-size 64

Produces, for each model trained:
  * outputs/checkpoints/<name>_best.pt   (best validation checkpoint, also
                                           copied to models/<name>.pt)
  * outputs/figures/<name>_training_curves.png (loss + accuracy curves)
  * outputs/metrics/<name>_history.json  (full per-epoch history)

The script never crashes with a bare traceback on a missing/empty dataset -
`DatasetNotFoundError` is caught and reported with instructions instead.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

if __name__ == "__main__" and not __package__:
    # Allow `python src/train.py ...` (run directly, not as `python -m`) by
    # putting the project root on sys.path so `from src import ...` resolves.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for saving figures without a display
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src import config
from src.cnn_model import build_model_a
from src.data_loader import (
    DatasetNotFoundError,
    build_frame_datasets,
    build_sequence_dataset,
    get_weighted_sampler,
)
from src.lstm_model import build_model_b
from src.utils import get_device, get_logger, save_checkpoint, set_seed

logger = get_logger(__name__)


def train_one_epoch(model, loader, optimizer, criterion, device) -> tuple[float, float]:
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        loss = criterion(logits, labels)
        running_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def plot_training_curves(history: dict, name: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history["train_loss"], label="Train Loss", marker="o", markersize=3)
    axes[0].plot(history["val_loss"], label="Validation Loss", marker="o", markersize=3)
    axes[0].set_title(f"{name} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train Accuracy", marker="o", markersize=3)
    axes[1].plot(history["val_acc"], label="Validation Accuracy", marker="o", markersize=3)
    axes[1].set_title(f"{name} - Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path = config.FIGURES_DIR / f"{name}_training_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved training curves -> {out_path}")
    return out_path


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    name: str,
    checkpoint_path: Path,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device: torch.device,
) -> dict:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.LR_SCHEDULER_FACTOR, patience=config.LR_SCHEDULER_PATIENCE
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    logger.info(f"[{name}] Training on device={device} for up to {epochs} epochs (patience={patience})")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        logger.info(
            f"[{name}] Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {elapsed:.1f}s"
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            save_checkpoint(
                {
                    "model_state_dict": best_state,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "config": {
                        "feature_dim": config.FEATURE_DIM,
                        "rnn_type": getattr(model, "rnn_type", None),
                        "sequence_length": config.SEQUENCE_LENGTH,
                    },
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(f"[{name}] Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    history_path = config.METRICS_DIR / f"{name}_history.json"
    history_path.write_text(json.dumps(history, indent=2))
    plot_training_curves(history, name)

    return history


def train_model_a(args) -> None:
    logger.info("=== Training Model A: CNN baseline (frame-level) ===")
    datasets = build_frame_datasets()
    sampler = get_weighted_sampler(datasets["train"])
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, sampler=sampler, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False, num_workers=config.NUM_WORKERS)

    logger.info(f"Class distribution (train): {datasets['train'].class_counts()}")

    # Default: fine-tune the whole backbone for Model A. Eye-crop close-ups
    # are visually far from ImageNet's natural-image domain, so letting the
    # CNN adapt its filters (not just the classifier head) matters more here
    # than for Model B, where we deliberately keep the backbone frozen.
    freeze_backbone = args.freeze_backbone if args.freeze_backbone is not None else False
    model = build_model_a(pretrained=not args.no_pretrained, freeze_backbone=freeze_backbone)
    device = get_device(config.DEVICE)

    run_training(
        model, train_loader, val_loader,
        name="cnn_baseline",
        checkpoint_path=config.CHECKPOINTS_DIR / "cnn_baseline_best.pt",
        epochs=args.epochs, lr=args.lr, weight_decay=config.WEIGHT_DECAY,
        patience=args.patience, device=device,
    )

    save_checkpoint({"model_state_dict": model.state_dict()}, config.CNN_CHECKPOINT_PATH)
    logger.info(f"Model A ready for evaluation/inference -> {config.CNN_CHECKPOINT_PATH}")


def train_model_b(args) -> None:
    logger.info("=== Training Model B: CNN + LSTM/GRU (proposed) ===")
    train_ds, val_ds, _test_ds = build_sequence_dataset(seq_len=args.sequence_length)
    train_loader = DataLoader(train_ds, batch_size=args.sequence_batch_size, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=args.sequence_batch_size, shuffle=False, num_workers=config.NUM_WORKERS)

    # Default: freeze the CNN backbone for Model B and only train the
    # RNN + classifier head. Sequence datasets are typically much smaller
    # than frame-level ones (fewer clips than individual frames), so keeping
    # the backbone fixed reduces overfitting risk and speeds up training.
    freeze_backbone = args.freeze_backbone if args.freeze_backbone is not None else True
    model = build_model_b(
        pretrained=not args.no_pretrained,
        freeze_backbone=freeze_backbone,
        rnn_type=args.rnn_type,
    )
    device = get_device(config.DEVICE)

    run_training(
        model, train_loader, val_loader,
        name="cnn_lstm_proposed",
        checkpoint_path=config.CHECKPOINTS_DIR / "cnn_lstm_proposed_best.pt",
        epochs=args.epochs, lr=args.lr, weight_decay=config.WEIGHT_DECAY,
        patience=args.patience, device=device,
    )

    save_checkpoint(
        {"model_state_dict": model.state_dict(), "rnn_type": args.rnn_type, "sequence_length": args.sequence_length},
        config.LSTM_CHECKPOINT_PATH,
    )
    logger.info(f"Model B ready for evaluation/inference -> {config.LSTM_CHECKPOINT_PATH}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the driver drowsiness detection models.")
    parser.add_argument("--model", choices=["cnn", "lstm", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--sequence-batch-size", type=int, default=config.SEQUENCE_BATCH_SIZE)
    parser.add_argument("--sequence-length", type=int, default=config.SEQUENCE_LENGTH)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=config.EARLY_STOPPING_PATIENCE)
    parser.add_argument("--rnn-type", choices=["GRU", "LSTM"], default=config.RNN_TYPE)
    parser.add_argument(
        "--freeze-backbone", dest="freeze_backbone", action="store_true", default=None,
        help="Force-freeze the CNN backbone. Default: False for Model A, True for Model B.",
    )
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet-pretrained MobileNetV2 weights.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    set_seed(config.RANDOM_SEED)

    try:
        if args.model in ("cnn", "both"):
            train_model_a(args)
        if args.model in ("lstm", "both"):
            train_model_b(args)
    except DatasetNotFoundError as exc:
        logger.error(f"Training aborted: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
