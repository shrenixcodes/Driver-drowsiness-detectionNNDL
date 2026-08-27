"""
Central configuration for the Driver Drowsiness Prediction project.

Every tunable constant lives here so behaviour can be changed in one place
instead of hunting through the codebase. The Streamlit app overrides a
subset of these at runtime (session-level only); it never rewrites this file.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"
SEQUENCE_DIR = DATA_DIR / "sequences"  # {drowsy,alert}/<video_name>/*.jpg

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
ASSETS_DIR = PROJECT_ROOT / "assets"

CNN_CHECKPOINT_PATH = MODELS_DIR / "cnn_baseline.pt"
LSTM_CHECKPOINT_PATH = MODELS_DIR / "cnn_lstm_proposed.pt"
ALARM_SOUND_PATH = ASSETS_DIR / "alarm.wav"

for _d in (MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR, METRICS_DIR, CHECKPOINTS_DIR, ASSETS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Class labels
# --------------------------------------------------------------------------
# Frame-level eye-state classes as found on disk (data/train/<class_name>/*.jpg)
EYE_STATE_CLASSES = ["closed_eye", "open_eye"]  # index 0 / 1 (alphabetical)
CLOSED_IDX = EYE_STATE_CLASSES.index("closed_eye")
OPEN_IDX = EYE_STATE_CLASSES.index("open_eye")

# Both models ultimately output a single scalar: P(drowsy). "closed_eye" is
# treated as the positive/drowsy-indicating class, "open_eye" as alert.
DROWSY_LABEL_NAMES = ["ALERT", "DROWSY"]

# --------------------------------------------------------------------------
# Image / preprocessing
# --------------------------------------------------------------------------
EYE_IMG_SIZE = 224          # MobileNetV2 expects 224x224 RGB input
FACE_IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --------------------------------------------------------------------------
# Temporal sequence (Model B)
# --------------------------------------------------------------------------
SEQUENCE_LENGTH = int(os.environ.get("DDL_SEQUENCE_LENGTH", 15))  # frames per sequence
SEQUENCE_STRIDE = 1          # step between consecutive sampled frames
FEATURE_DIM = 128            # dimensionality of CNN feature vector fed to the RNN
RNN_TYPE = "GRU"             # "GRU" or "LSTM" - both implemented, GRU is the default
RNN_HIDDEN_SIZE = 64
RNN_NUM_LAYERS = 1
RNN_DROPOUT = 0.3

# --------------------------------------------------------------------------
# Training hyperparameters
# --------------------------------------------------------------------------
BATCH_SIZE = 32
SEQUENCE_BATCH_SIZE = 16
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 6
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_FACTOR = 0.5
VAL_SPLIT = 0.15   # only used when a dataset provides no explicit validation folder
TEST_SPLIT = 0.15  # only used when a dataset provides no explicit test folder
NUM_WORKERS = 0    # Windows + webcam workloads: keep DataLoader workers at 0 by default
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Real-time drowsiness decision logic
# --------------------------------------------------------------------------
# Probability thresholds (configurable from the Streamlit sidebar too)
DROWSY_THRESHOLD = 0.60      # >= this => DROWSY
HIGH_RISK_THRESHOLD = 0.80   # >= this => HIGH RISK

# Temporal smoothing / debouncing so a single bad frame never triggers a warning
SMOOTHING_WINDOW = 10          # frames averaged for the displayed probability
WARNING_DURATION = 15          # consecutive high-risk *smoothed* frames required to sound the alarm
EYE_CLOSED_CONSEC_FRAMES = 20  # consecutive closed-eye frames considered "prolonged closure"

# Classical Eye Aspect Ratio (EAR) / Mouth Aspect Ratio (MAR) thresholds,
# used both as auxiliary signals and as the standalone "heuristic mode"
# that requires no trained model at all.
EAR_THRESHOLD = 0.21
MAR_YAWN_THRESHOLD = 0.60
BLINK_MIN_CONSEC_FRAMES = 2   # frames below EAR_THRESHOLD to count as one blink

# --------------------------------------------------------------------------
# Face / landmark detection
# --------------------------------------------------------------------------
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5
HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBORS = 5

# --------------------------------------------------------------------------
# Webcam / app
# --------------------------------------------------------------------------
DEFAULT_CAMERA_INDEX = 0
TARGET_FPS = 30

DEVICE = os.environ.get("DDL_DEVICE", "auto")  # "auto" | "cpu" | "cuda"

# Academic-integrity / safety disclaimer shown in the app and README
DISCLAIMER = (
    "This is an academic prototype built for a Neural Networks and Deep "
    "Learning coursework project. It is NOT a certified automotive safety "
    "system and must not be relied upon for real-world driving safety."
)
