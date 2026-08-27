# Spatiotemporal Deep Learning for Real-Time Driver Drowsiness Prediction and Early Warning

An academic Neural Networks and Deep Learning course project: a webcam-based application that
monitors a driver's face in real time and predicts drowsiness, using **temporal** deep learning
(CNN + LSTM/GRU) rather than classifying individual frames in isolation.

> **This is an academic prototype, not a certified automotive safety system.** It must not be
> relied upon for real-world driving safety. See [Limitations](#13-limitations).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Motivation](#3-motivation)
4. [System Architecture](#4-system-architecture)
5. [Dataset](#5-dataset)
6. [Installation](#6-installation)
7. [Dataset Preparation](#7-dataset-preparation)
8. [Training](#8-training)
9. [Evaluation](#9-evaluation)
10. [Running the Real-Time Application](#10-running-the-real-time-application)
11. [Model Architecture](#11-model-architecture)
12. [Results](#12-results)
13. [Limitations](#13-limitations)
14. [Future Improvements](#14-future-improvements)

---

## 1. Project Overview

The system watches a driver through a laptop webcam, tracks eye/mouth state and head pose,
and predicts a drowsiness probability every frame. Two models are implemented and compared:

- **Model A (baseline):** a CNN classifies each frame's eye crop independently as ALERT or DROWSY.
- **Model B (proposed):** the same CNN backbone extracts a feature vector per frame, and a
  GRU/LSTM consumes a rolling sequence of those vectors to make a temporally-aware prediction.

The project's core academic question is: **does adding temporal modeling (Model B) improve
drowsiness prediction over a frame-by-frame baseline (Model A)?** `src/evaluate.py` answers this
with real, measured numbers (accuracy/precision/recall/F1/latency) - nothing in the comparison
table is fabricated; a model that hasn't been trained/evaluated simply doesn't appear.

## 2. Problem Statement

Driver drowsiness is a major contributor to road accidents. Simple eye-closure detectors react to
a single frame and are prone to false alarms (a blink looks identical to the start of a
microsleep in one frame) and false negatives (a driver can keep their eyes technically "open"
while cognitively fading). A system that reasons over a **window of time** - blink duration,
closure frequency, gradual drift - is closer to how the danger actually manifests.

## 3. Motivation

Most classroom drowsiness-detection demos stop at "if eyes closed for N frames, alarm." This
project instead builds a full spatiotemporal pipeline (CNN feature extraction feeding a
recurrent model), trains it properly with a documented data pipeline, and **quantifies** whether
the added temporal machinery is worth its complexity versus the naive baseline - the point isn't
just to build an alarm, it's to demonstrate the value (or limits) of temporal deep learning for
this task.

## 4. System Architecture

```
Webcam
  │
  ▼
Face Detection & Landmarks (MediaPipe Face Mesh, Haar-cascade fallback)
  │  ├── Eye crop(s)              ──► EAR (Eye Aspect Ratio)
  │  ├── Mouth landmarks          ──► MAR (Mouth Aspect Ratio) → yawn signal
  │  └── 6-point solvePnP         ──► head pitch/yaw/roll
  ▼
CNN Feature Extraction (MobileNetV2, transfer learning)
  │
  ├── Model A: CNN → Dense → ALERT / DROWSY               (single frame)
  │
  └── Model B: CNN → feature vector → rolling sequence of N frames
                → GRU/LSTM → Dense → P(drowsy)             (temporal)
  ▼
Drowsiness Logic (src/drowsiness_logic.py)
  raw probability → rolling-average smoothing → threshold
  → consecutive-high-risk-frame counter → debounced warning
  ▼
Streamlit Dashboard (app.py): live feed, metrics, history chart, audio alarm
```

**Why MobileNetV2?** Purpose-built for mobile/CPU inference (depthwise-separable convolutions,
~3.5M params), pretrained on ImageNet for transfer learning, and light enough for real-time
webcam inference on a normal laptop CPU.

**Why GRU by default (LSTM is a one-line config change)?** GRU has ~25% fewer parameters than an
LSTM of the same hidden size (no separate cell state), which lowers per-frame inference latency -
directly relevant to the real-time constraint. At the short sequence lengths used here, GRU and
LSTM perform comparably, so the latency edge dominates the choice. Switch via `RNN_TYPE` in
`src/config.py` or `--rnn-type LSTM` on `train.py`.

**Why MediaPipe (with an OpenCV Haar-cascade fallback)?** MediaPipe installs via plain `pip`
with no compiler toolchain, unlike `dlib`, which typically needs CMake/Visual Studio build tools
on Windows. If MediaPipe fails to import or initialize, `src/face_detection.py` automatically
falls back to Haar cascades bundled with `opencv-python`, so the app degrades gracefully instead
of crashing (with reduced signal quality - see [Limitations](#13-limitations)).

## 5. Dataset

### Primary dataset (frame-level, Model A + CNN feature training): MRL Eye Dataset

The [MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset) is a large, well-established collection of
open/closed eye images, commonly mirrored on Kaggle (search "MRL Eye Dataset"). It is free for
research/educational use; check the current license terms on the source page before use, and
cite the dataset in any coursework write-up.

Download it and arrange it as:

```
data/
├── train/
│   ├── open_eye/*.jpg
│   └── closed_eye/*.jpg
├── validation/          # optional - see note below
│   ├── open_eye/*.jpg
│   └── closed_eye/*.jpg
└── test/                # optional - see note below
    ├── open_eye/*.jpg
    └── closed_eye/*.jpg
```

If you only populate `data/train/`, `src/data_loader.py` automatically carves out stratified
validation/test splits from it (`VAL_SPLIT`/`TEST_SPLIT` in `src/config.py`) - you do not have to
split the dataset by hand.

### Sequence dataset (temporal, Model B): NTHU-DDD or YawDD

Model B needs **genuine video sequences** labeled drowsy/alert to learn real temporal dynamics.
The two standard datasets for this are:

- **NTHU Drowsy Driver Detection (NTHU-DDD)** - request access from the NTHU Computer Vision Lab
  (search "NTHU DDD dataset access request"). Access is gated behind an academic-use agreement
  with the original authors; this project cannot and does not provide a direct download link.
- **YawDD** - similarly requires requesting access from the dataset authors.

Once you have one of these (or any similarly-labeled video dataset), extract per-frame eye crops
in temporal order with the provided helper:

```python
from src.preprocessing import extract_eye_sequence_from_video

extract_eye_sequence_from_video(
    video_path="path/to/clip_01.mp4",
    out_dir="data/sequences/drowsy/clip_01",   # or data/sequences/alert/clip_01
)
```

Repeat for every clip, sorting each into `data/sequences/drowsy/<clip_name>/` or
`data/sequences/alert/<clip_name>/`. `src/data_loader.py` then builds sliding-window sequences of
`SEQUENCE_LENGTH` frames automatically.

**No video dataset yet?** The pipeline still runs end-to-end: `src/data_loader.py` falls back to
building *pseudo-sequences* from the MRL eye-state images (grouped by the subject ID encoded in
MRL filenames, e.g. `s0001_...`). This is clearly logged at training time and exists **only** so
Model B's code path is fully runnable for demonstration - it contains no genuine temporal
dynamics (the frames are not consecutive video frames), and results from this mode should not be
presented as evidence of temporal-modeling quality. See [Limitations](#13-limitations).

## 6. Installation

Requires Python 3.9+ (CPU is fully supported; CUDA is used automatically if available).

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 7. Dataset Preparation

1. Download the MRL Eye Dataset (see [Dataset](#5-dataset)) and place images under
   `data/train/open_eye/` and `data/train/closed_eye/`.
2. (Optional) Add `data/validation/` and `data/test/` in the same layout, or let the training
   script auto-split `data/train/` for you.
3. (Optional, for Model B) Obtain NTHU-DDD or YawDD, then run
   `extract_eye_sequence_from_video` (see above) for each clip to populate
   `data/sequences/{drowsy,alert}/<clip_name>/`.

## 8. Training

```bash
# Train both models sequentially
python src/train.py --model both

# Or train one at a time
python src/train.py --model cnn
python src/train.py --model lstm --rnn-type GRU --sequence-length 15

# Useful flags
python src/train.py --model cnn --epochs 15 --batch-size 64 --lr 1e-4
```

Each run performs: stratified train/val/test split (auto-derived if needed), data augmentation,
class-balanced sampling, early stopping, LR-plateau scheduling, and checkpointing of the
best-validation-loss model. Outputs land under `outputs/`:

- `outputs/figures/<model>_training_curves.png` - loss & accuracy curves
- `outputs/metrics/<model>_history.json` - full per-epoch history
- `outputs/checkpoints/<model>_best.pt` + `models/<model>.pt` - the trained weights used by
  evaluation and real-time inference

If `data/train/` is empty, training aborts immediately with a clear message explaining exactly
what to place where - it does not crash with a raw stack trace.

## 9. Evaluation

```bash
python src/evaluate.py --model both
```

Computes accuracy, precision, recall, F1, a confusion matrix, and single-frame inference latency
for each trained model on its held-out test split, and writes:

- `outputs/figures/<model>_confusion_matrix.png`
- `outputs/metrics/<model>_classification_report.txt`
- `outputs/metrics/comparison_table.csv` / `.json` - read directly by the Streamlit dashboard's
  "Model Comparison" tab

A model with no checkpoint yet is skipped with a warning rather than failing the whole run - the
comparison table only ever contains models that were actually evaluated.

## 10. Running the Real-Time Application

**Streamlit dashboard (recommended):**

```bash
streamlit run app.py
```

Configure the model, thresholds, sequence length, and camera in the sidebar, then click **Start**.
Works even with no trained model at all via **Heuristic (EAR/MAR)** mode, so the demo is always
runnable.

**Standalone OpenCV window (no browser required):**

```bash
python src/inference.py --model cnn_lstm --camera 0
```

`--model` accepts `cnn_lstm`, `cnn`, or `heuristic`. If the requested model's checkpoint is
missing, inference automatically falls back (`cnn_lstm` → `cnn` → `heuristic`) and logs why.

## 11. Model Architecture

```
Input frame (eye crop, 224x224 RGB)
        │
        ▼
MobileNetV2 conv trunk (ImageNet-pretrained)
        │
        ▼
Global Average Pool → Linear(1280 → 128) → ReLU → Dropout   [MobileNetFeatureExtractor]
        │
        ├─────────────────────────────┐
        ▼                             ▼
  Model A: Dense(128→32)→Dense(32→2)   Model B: stack 128-d features over
  → ALERT / DROWSY (single frame)      SEQUENCE_LENGTH frames → GRU/LSTM
                                        (hidden=64) → Dense(64→32)→Dense(32→2)
                                        → ALERT / DROWSY (temporal)
```

Model A fine-tunes the full MobileNetV2 backbone by default (eye-crop close-ups are visually far
from ImageNet's natural-image domain, so adapting the filters helps). Model B freezes the
backbone by default and only trains the GRU/LSTM + classifier head, since sequence datasets are
typically much smaller than frame-level ones - both defaults are overridable via
`--freeze-backbone` / `--no-freeze-backbone` on `train.py`.

## 12. Results

| Metric | CNN (Baseline) | CNN + LSTM/GRU (Proposed) |
|---|---:|---:|
| Accuracy | *run `python src/evaluate.py`* | *run `python src/evaluate.py`* |
| Precision | | |
| Recall | | |
| F1 Score | | |
| Inference Time | | |

This table is intentionally left blank in source control - it is generated by
`src/evaluate.py` from your actual trained models and dataset, and is also rendered live (with
confusion matrices and training curves) in the Streamlit app's **Model Comparison** tab. Do not
hand-fill this table with invented numbers.

## 13. Limitations

- **Not a certified safety system.** This is a coursework prototype; it has not been validated
  against automotive safety standards and must never be relied upon while actually driving.
- **Pseudo-sequence fallback.** Without a genuine video dataset (NTHU-DDD/YawDD), Model B trains
  on pseudo-sequences built from the frame-level eye dataset, which do not contain real temporal
  dynamics. Treat any results from this mode as a demonstration of the *code path*, not of
  temporal-modeling quality.
- **Haar-cascade fallback is coarser.** If MediaPipe is unavailable, the Haar-cascade backend
  only supplies bounding boxes; EAR is approximated from eye-box aspect ratio (not true eyelid
  landmarks), and MAR/head-pose are unavailable in that mode.
- **MediaPipe API churn.** Some recent MediaPipe releases removed the legacy `mp.solutions`
  Python API this project's primary backend uses (in favor of a newer Tasks API that requires a
  separately downloaded model file). If your installed `mediapipe` lacks `mp.solutions`,
  `face_detection.py` detects this automatically and falls back to the bundled Haar cascades
  (`assets/haarcascades/`) - the app keeps working, just with the reduced signal quality noted
  above. If you want full landmark-based EAR/MAR/head-pose, install a `mediapipe` version that
  still ships `mp.solutions.face_mesh` (e.g. pin `mediapipe==0.10.14` in `requirements.txt`).
- **Single-face assumption.** The pipeline tracks one face at a time (the driver), matching the
  intended use case.
- **Lighting/camera sensitivity.** Like any webcam-based CV system, performance depends on
  adequate lighting and a front-facing camera angle.
- **EAR/MAR thresholds are empirical defaults**, tuned to commonly-cited literature values, not
  calibrated per-individual - the sidebar lets you adjust them per session.

## 14. Future Improvements

- Fine-tune EAR/MAR thresholds per-user via a short on-device calibration step.
- Multi-modal fusion of eye state, yawn, and head-pose signals with learned (rather than
  rule-based) weighting.
- Attention-based temporal models (e.g. a small Transformer) as an alternative to GRU/LSTM.
- On-device quantization (e.g. `torch.quantization`) for even lower CPU latency.
- Proper video-sequence dataset support once NTHU-DDD/YawDD access is obtained, replacing the
  pseudo-sequence fallback entirely.

---

## Project Structure

```
driver-drowsiness-dl/
├── app.py                     # Streamlit dashboard
├── requirements.txt
├── README.md
├── data/                      # train/validation/test/sequences (user-provided)
├── models/                    # final trained checkpoints (cnn_baseline.pt, cnn_lstm_proposed.pt)
├── src/
│   ├── config.py              # all tunable constants (thresholds, SEQUENCE_LENGTH, paths, ...)
│   ├── utils.py                # logging, device, checkpoints, alarm sound
│   ├── face_detection.py      # MediaPipe / Haar face+eye+mouth detection, EAR/MAR, head pose
│   ├── preprocessing.py       # image transforms, video → sequence extraction
│   ├── data_loader.py         # frame-level & sequence PyTorch datasets
│   ├── cnn_model.py            # Model A (MobileNetV2 CNN baseline)
│   ├── lstm_model.py           # Model B (MobileNetV2 + GRU/LSTM)
│   ├── train.py                 # unified training pipeline
│   ├── evaluate.py              # metrics, confusion matrices, comparison table
│   ├── drowsiness_logic.py    # smoothing, debouncing, state machine
│   ├── explainability.py      # Grad-CAM for Model A
│   └── inference.py           # real-time engine + standalone OpenCV demo
├── notebooks/                 # dataset exploration notebook
├── outputs/
│   ├── figures/                # training curves, confusion matrices
│   ├── metrics/                 # history JSON, classification reports, comparison table
│   └── checkpoints/            # best-epoch checkpoints during training
└── assets/
    └── alarm.wav                # locally-synthesized alarm tone (no external audio asset)
```
