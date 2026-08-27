"""
Shared utilities: logging, reproducibility, device selection, checkpoint
save/load helpers, and the audible alarm (synthesized locally, no external
audio asset is downloaded from anywhere).
"""
from __future__ import annotations

import logging
import math
import random
import sys
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard requirement, but fail softly on import
    torch = None

from src import config


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that prints to stdout without duplicate handlers."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


logger = get_logger("ddl")


# --------------------------------------------------------------------------
# Reproducibility / device
# --------------------------------------------------------------------------
def set_seed(seed: int = config.RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto") -> "torch.device":
    """Resolve the compute device, defaulting to CPU when CUDA isn't usable.

    The whole project is designed to run comfortably on CPU; CUDA is used
    opportunistically when available and requested.
    """
    if torch is None:
        raise RuntimeError("PyTorch is not installed. Run: pip install -r requirements.txt")

    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available - falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")

    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------
def save_checkpoint(state: dict, path: Path) -> None:
    if torch is None:
        raise RuntimeError("PyTorch is not installed.")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    logger.info(f"Saved checkpoint -> {path}")


def load_checkpoint(path: Path, map_location: Optional[str] = "cpu") -> Optional[dict]:
    """Load a checkpoint dict, returning None (not raising) if the file is missing.

    This lets every downstream consumer (inference, evaluate, Streamlit app)
    degrade gracefully with a clear user-facing message instead of crashing.
    """
    if torch is None:
        raise RuntimeError("PyTorch is not installed.")
    path = Path(path)
    if not path.exists():
        logger.warning(f"Checkpoint not found: {path}")
        return None
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:  # noqa: BLE001 - surface any load error clearly
        logger.error(f"Failed to load checkpoint {path}: {exc}")
        return None


def checkpoint_exists(path: Path) -> bool:
    return Path(path).exists()


# --------------------------------------------------------------------------
# Timing helper (used for latency benchmarking)
# --------------------------------------------------------------------------
class Timer:
    """Millisecond-precision context manager: `with Timer() as t: ...; t.elapsed_ms`."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


# --------------------------------------------------------------------------
# Alarm sound: synthesized locally so no audio file needs to be sourced
# --------------------------------------------------------------------------
def synthesize_alarm_wav(
    path: Path = config.ALARM_SOUND_PATH,
    duration_s: float = 1.2,
    freq_hz: float = 880.0,
    sample_rate: int = 44100,
) -> Path:
    """Generate a simple two-tone beep WAV file and write it to `path`.

    Idempotent - if the file already exists it is left untouched. This keeps
    the project free of any downloaded/copyrighted media asset.
    """
    path = Path(path)
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(duration_s * sample_rate)
    samples = np.zeros(n_samples, dtype=np.float32)

    # Two alternating tones (like a classic alert beep) with a short envelope
    # to avoid audible clicking at segment boundaries.
    segment = n_samples // 2
    t1 = np.linspace(0, duration_s / 2, segment, endpoint=False)
    t2 = np.linspace(0, duration_s / 2, n_samples - segment, endpoint=False)
    tone1 = 0.6 * np.sin(2 * math.pi * freq_hz * t1)
    tone2 = 0.6 * np.sin(2 * math.pi * (freq_hz * 1.25) * t2)

    envelope1 = np.minimum(1.0, np.minimum(t1 * 50, (t1[-1] - t1) * 50 + 0.05)) if segment > 0 else t1
    envelope2 = np.minimum(1.0, np.minimum(t2 * 50, (t2[-1] - t2) * 50 + 0.05)) if len(t2) > 0 else t2

    samples[:segment] = tone1 * envelope1
    samples[segment:] = tone2 * envelope2

    pcm = np.int16(np.clip(samples, -1.0, 1.0) * 32767)

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    logger.info(f"Synthesized alarm sound -> {path}")
    return path


class AlarmPlayer:
    """Plays the alarm WAV via pygame.mixer without blocking the main loop.

    Gracefully no-ops if pygame's audio backend is unavailable (e.g. a
    headless CI machine with no sound device) - the visual warning still
    works even if audio does not.
    """

    def __init__(self, sound_path: Path = config.ALARM_SOUND_PATH):
        self.enabled = True
        self._sound = None
        self._mixer = None
        try:
            synthesize_alarm_wav(sound_path)
            import pygame

            pygame.mixer.init()
            self._mixer = pygame.mixer
            self._sound = pygame.mixer.Sound(str(sound_path))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Audio disabled (no output device or pygame unavailable): {exc}")
            self.enabled = False

    def play(self) -> None:
        if not self.enabled or self._sound is None:
            return
        try:
            if not self._mixer.get_busy():
                self._sound.play()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to play alarm: {exc}")

    def stop(self) -> None:
        if not self.enabled or self._sound is None:
            return
        try:
            self._sound.stop()
        except Exception:  # noqa: BLE001
            pass
