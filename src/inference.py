"""
Real-time inference engine + standalone webcam demo.

`DrowsinessInferenceEngine` is the single place that ties together face/eye
detection, the CNN or CNN+LSTM model (or a model-free EAR/MAR heuristic),
and the temporal smoothing/debouncing logic. It is used both by this file's
`main()` (a plain OpenCV window, run via `python src/inference.py`) and by
`app.py` (the Streamlit dashboard) so the two front-ends never duplicate the
core logic.

Model selection cascades gracefully: if the requested model's checkpoint is
missing, the engine falls back to a simpler mode (cnn_lstm -> cnn ->
heuristic) and reports why via `status_message`, instead of crashing.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

from src import config
from src.cnn_model import DrowsinessCNN, build_model_a
from src.drowsiness_logic import DrowsinessLogic, DrowsinessState, heuristic_probability_from_signals
from src.face_detection import FaceDetectionResult, FaceLandmarkDetector
from src.lstm_model import CNNRNNDrowsinessModel, build_model_b
from src.preprocessing import preprocess_frame_for_model
from src.utils import AlarmPlayer, get_device, get_logger, load_checkpoint

logger = get_logger(__name__)

_STATUS_COLORS_BGR = {
    "ALERT": (0, 200, 0),
    "DROWSY": (0, 165, 255),
    "HIGH_RISK": (0, 0, 255),
}


@dataclass
class InferenceResult:
    frame: np.ndarray
    detection: FaceDetectionResult
    state: DrowsinessState
    fps: float
    latency_ms: float
    active_mode: str


class DrowsinessInferenceEngine:
    def __init__(
        self,
        model_type: str = "cnn_lstm",
        sequence_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        drowsy_threshold: Optional[float] = None,
        high_risk_threshold: Optional[float] = None,
        warning_duration: Optional[int] = None,
        smoothing_window: Optional[int] = None,
        enable_audio: bool = True,
        face_backend: str = "auto",
    ):
        self.sequence_length = sequence_length or config.SEQUENCE_LENGTH
        self.device = device or get_device(config.DEVICE)
        self.detector = FaceLandmarkDetector(backend=face_backend)
        self.logic = DrowsinessLogic(
            drowsy_threshold=drowsy_threshold or config.DROWSY_THRESHOLD,
            high_risk_threshold=high_risk_threshold or config.HIGH_RISK_THRESHOLD,
            warning_duration=warning_duration or config.WARNING_DURATION,
            smoothing_window=smoothing_window or config.SMOOTHING_WINDOW,
        )
        self.alarm = AlarmPlayer() if enable_audio else None
        self.feature_buffer: deque = deque(maxlen=self.sequence_length)

        self.cnn_model: Optional[DrowsinessCNN] = None
        self.lstm_model: Optional[CNNRNNDrowsinessModel] = None
        self.active_mode, self.status_message = self._load_models(model_type)
        if self.status_message:
            logger.warning(self.status_message)

    # ------------------------------------------------------------------
    def _try_load_cnn(self) -> Optional[DrowsinessCNN]:
        ckpt = load_checkpoint(config.CNN_CHECKPOINT_PATH)
        if ckpt is None:
            return None
        try:
            model = build_model_a(pretrained=False)
            model.load_state_dict(ckpt["model_state_dict"])
            return model.to(self.device).eval()
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to initialize CNN model from checkpoint: {exc}")
            return None

    def _try_load_lstm(self) -> Optional[CNNRNNDrowsinessModel]:
        ckpt = load_checkpoint(config.LSTM_CHECKPOINT_PATH)
        if ckpt is None:
            return None
        try:
            rnn_type = ckpt.get("rnn_type", config.RNN_TYPE)
            seq_len = ckpt.get("sequence_length")
            if seq_len:
                self.sequence_length = seq_len
                self.feature_buffer = deque(maxlen=self.sequence_length)
            model = build_model_b(pretrained=False, freeze_backbone=True, rnn_type=rnn_type)
            model.load_state_dict(ckpt["model_state_dict"])
            return model.to(self.device).eval()
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to initialize CNN+LSTM model from checkpoint: {exc}")
            return None

    def _load_models(self, requested: str) -> tuple[str, Optional[str]]:
        warnings = []

        if requested == "cnn_lstm":
            model = self._try_load_lstm()
            if model is not None:
                self.lstm_model = model
                return "cnn_lstm", None
            warnings.append(f"No CNN+LSTM checkpoint found at '{config.LSTM_CHECKPOINT_PATH}'.")
            requested = "cnn"  # cascade down

        if requested == "cnn":
            model = self._try_load_cnn()
            if model is not None:
                self.cnn_model = model
                msg = (" ".join(warnings) + " Falling back to the CNN baseline.") if warnings else None
                return "cnn", msg
            warnings.append(f"No CNN checkpoint found at '{config.CNN_CHECKPOINT_PATH}'.")

        if warnings:
            msg = " ".join(warnings) + (
                " Falling back to heuristic (EAR/MAR-based) mode - train a model "
                "with 'python src/train.py' to enable deep-learning inference."
            )
        else:
            msg = "Running in heuristic (EAR/MAR-based) mode - no trained model required."
        return "heuristic", msg

    # ------------------------------------------------------------------
    def _compute_probability(self, crop: Optional[np.ndarray], detection: FaceDetectionResult) -> float:
        if crop is None or self.active_mode == "heuristic":
            return heuristic_probability_from_signals(detection.ear, self.logic.consecutive_closed_frames)

        tensor = preprocess_frame_for_model(crop).unsqueeze(0).to(self.device)

        if self.active_mode == "cnn_lstm":
            with torch.no_grad():
                feature = self.lstm_model.extract_features(tensor)
            self.feature_buffer.append(feature.squeeze(0).cpu())

            if len(self.feature_buffer) < self.sequence_length:
                # Not enough history yet to run the temporal model - use the
                # heuristic as a safe bridge during the first ~N frames.
                return heuristic_probability_from_signals(detection.ear, self.logic.consecutive_closed_frames)

            seq = torch.stack(list(self.feature_buffer), dim=0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                prob = self.lstm_model.predict_proba_from_features(seq).item()
            return prob

        # active_mode == "cnn"
        with torch.no_grad():
            prob = self.cnn_model.predict_proba(tensor).item()
        return prob

    # ------------------------------------------------------------------
    def process_frame(self, frame_bgr: np.ndarray, overlay_style: str = "boxes_only") -> InferenceResult:
        t0 = time.perf_counter()
        detection = self.detector.process(frame_bgr)

        if not detection.success:
            state = self.logic.update(raw_probability=0.0, ear=None, mar=None)
        else:
            crop = detection.left_eye_crop if detection.left_eye_crop is not None else detection.right_eye_crop
            probability = self._compute_probability(crop, detection)
            state = self.logic.update(probability, ear=detection.ear, mar=detection.mar)

        if self.alarm is not None:
            if state.warning_active:
                self.alarm.play()
            else:
                self.alarm.stop()

        latency_ms = (time.perf_counter() - t0) * 1000.0
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

        annotated = self._draw_overlay(frame_bgr.copy(), detection, state, fps, overlay_style)

        return InferenceResult(
            frame=annotated, detection=detection, state=state, fps=fps, latency_ms=latency_ms,
            active_mode=self.active_mode,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _draw_overlay(frame, detection: FaceDetectionResult, state: DrowsinessState, fps: float, style: str):
        color = _STATUS_COLORS_BGR.get(state.status, (255, 255, 255))

        if detection.success and detection.face_bbox:
            x, y, w, h = detection.face_bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        for bbox in (detection.left_eye_bbox, detection.right_eye_bbox):
            if bbox:
                ex, ey, ew, eh = bbox
                cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 200, 0), 1)
        if detection.mouth_bbox:
            mx, my, mw, mh = detection.mouth_bbox
            cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (0, 255, 255), 1)

        if style == "full":
            lines = [
                "DRIVER MONITORING SYSTEM",
                f"Status: {state.status}",
                f"Drowsiness Probability: {state.smoothed_probability * 100:.0f}%",
                f"Eye State: {state.eye_state}",
                f"Blink Rate: {state.blink_rate_per_min:.0f}/min",
                f"FPS: {fps:.0f}",
                f"System Status: {'WARNING' if state.warning_active else 'NORMAL'}",
            ]
            for i, line in enumerate(lines):
                y_pos = 25 + i * 24
                weight = 2 if i in (0, 1) else 1
                cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color if i == 1 else (255, 255, 255), weight, cv2.LINE_AA)

            if state.warning_active:
                banner = "!! DROWSINESS DETECTED - TAKE A BREAK !!"
                cv2.putText(frame, banner, (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            if not detection.success:
                cv2.putText(frame, "No face detected", (10, frame.shape[0] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        return frame

    def close(self) -> None:
        self.detector.close()
        if self.alarm is not None:
            self.alarm.stop()


def list_available_cameras(max_index: int = 5) -> list:
    """Best-effort probe of camera indices 0..max_index-1. Used by the
    Streamlit sidebar's camera-selection dropdown. Never raises - an
    inaccessible index is simply omitted from the result.
    """
    available = []
    for idx in range(max_index):
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(idx)
            if cap is not None and cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    available.append(idx)
            cap.release()
        except Exception:  # noqa: BLE001
            continue
    return available or [config.DEFAULT_CAMERA_INDEX]


def main():
    parser = argparse.ArgumentParser(description="Real-time driver drowsiness monitoring (standalone OpenCV window).")
    parser.add_argument("--model", choices=["cnn_lstm", "cnn", "heuristic"], default="cnn_lstm")
    parser.add_argument("--camera", type=int, default=config.DEFAULT_CAMERA_INDEX)
    parser.add_argument("--sequence-length", type=int, default=config.SEQUENCE_LENGTH)
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args()

    engine = DrowsinessInferenceEngine(
        model_type=args.model, sequence_length=args.sequence_length, enable_audio=not args.no_audio,
    )
    logger.info(f"Active inference mode: {engine.active_mode}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error(
            f"Could not open webcam at index {args.camera}. Check that a camera is connected, not in use by "
            "another application, and that OS camera permissions are granted. Try a different --camera index."
        )
        return

    logger.info("Webcam opened. Press 'q' in the video window to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.error("Failed to read a frame from the webcam - stopping.")
                break
            frame = cv2.flip(frame, 1)  # mirror for a natural "looking in a mirror" view
            result = engine.process_frame(frame, overlay_style="full")
            cv2.imshow("Driver Drowsiness Monitoring - Press Q to Quit", result.frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        engine.close()


if __name__ == "__main__":
    main()
