"""
Face / eye / mouth landmark detection.

Primary backend: MediaPipe Face Mesh (468 3D landmarks, CPU, pip-installable
with no compiler toolchain required). This gives precise eye and mouth crops
plus the points needed for Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR),
and head-pose (pitch/yaw/roll) estimation.

Fallback backend: OpenCV Haar cascades (bundled with opencv-python, so it
never requires an extra download). It only yields bounding boxes, not
landmarks, so EAR/MAR/head-pose are approximated or unavailable - this is
documented and surfaced to the caller via `FaceDetectionResult.backend`.

The detector is intentionally backend-agnostic from the caller's point of
view: `FaceLandmarkDetector().process(frame)` always returns the same
`FaceDetectionResult` shape, so the rest of the pipeline (drowsiness logic,
inference loop, Streamlit app) does not need to know which backend is active.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from src import config
from src.utils import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------
# MediaPipe Face Mesh landmark index groups
# (indices into the 468-point face mesh; see MediaPipe's canonical face mesh
# topology - these are the standard points used across the EAR/MAR literature)
# --------------------------------------------------------------------------
LEFT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]   # p1..p6 for the EAR formula
RIGHT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_OUTLINE_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_OUTLINE_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308

# 6-point subset used for solvePnP head-pose estimation
HEAD_POSE_IDX = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_corner": 33,
    "right_eye_corner": 263,
    "left_mouth_corner": 61,
    "right_mouth_corner": 291,
}
# Generic 3D face model (mm) matched to the indices above - standard values
# used throughout the OpenCV head-pose-estimation literature/tutorials.
_MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),  # nose tip
        (0.0, -330.0, -65.0),  # chin
        (-225.0, 170.0, -135.0),  # left eye left corner
        (225.0, 170.0, -135.0),  # right eye right corner
        (-150.0, -150.0, -125.0),  # left mouth corner
        (150.0, -150.0, -125.0),  # right mouth corner
    ],
    dtype=np.float64,
)


@dataclass
class FaceDetectionResult:
    success: bool
    backend: str = "none"
    face_bbox: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h (pixels)
    left_eye_crop: Optional[np.ndarray] = None
    right_eye_crop: Optional[np.ndarray] = None
    left_eye_bbox: Optional[Tuple[int, int, int, int]] = None
    right_eye_bbox: Optional[Tuple[int, int, int, int]] = None
    mouth_bbox: Optional[Tuple[int, int, int, int]] = None
    ear: Optional[float] = None          # Eye Aspect Ratio, averaged over both eyes
    mar: Optional[float] = None          # Mouth Aspect Ratio
    head_pitch_deg: Optional[float] = None
    head_yaw_deg: Optional[float] = None
    head_roll_deg: Optional[float] = None
    landmarks_px: Optional[np.ndarray] = None  # (468, 2) if mediapipe backend


def _euclidean(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


def _compute_ear(pts: np.ndarray, idx: list) -> float:
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
    vertical = _euclidean(p2, p6) + _euclidean(p3, p5)
    horizontal = 2.0 * _euclidean(p1, p4)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def _compute_mar(pts: np.ndarray) -> float:
    vertical = _euclidean(pts[MOUTH_TOP], pts[MOUTH_BOTTOM])
    horizontal = _euclidean(pts[MOUTH_LEFT], pts[MOUTH_RIGHT])
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def _bbox_from_points(pts: np.ndarray, margin: float, w: int, h: int) -> Tuple[int, int, int, int]:
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    bw, bh = x_max - x_min, y_max - y_min
    x_min -= bw * margin
    x_max += bw * margin
    y_min -= bh * margin
    y_max += bh * margin
    x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
    x_max, y_max = min(w, int(x_max)), min(h, int(y_max))
    return x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min)


def crop_region(frame: np.ndarray, bbox: Tuple[int, int, int, int], out_size: int) -> np.ndarray:
    x, y, w, h = bbox
    crop = frame[y : y + h, x : x + w]
    if crop.size == 0:
        crop = np.zeros((out_size, out_size, 3), dtype=np.uint8)
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


class FaceLandmarkDetector:
    """Detects the driver's face and extracts eye/mouth regions + signals.

    Parameters
    ----------
    backend: "auto" | "mediapipe" | "haar"
        "auto" tries MediaPipe first and transparently falls back to Haar
        cascades if MediaPipe cannot be imported/initialized on this machine.
    """

    def __init__(self, backend: str = "auto"):
        self.backend = "none"
        self._mp_face_mesh = None
        self._haar_face = None
        self._haar_eye = None

        if backend in ("auto", "mediapipe"):
            self._init_mediapipe()

        if self.backend == "none" and backend in ("auto", "haar"):
            self._init_haar()

        if self.backend == "none":
            raise RuntimeError(
                "No face-detection backend could be initialized. Install either "
                "'mediapipe' or ensure OpenCV's bundled Haar cascade files are present."
            )

        logger.info(f"FaceLandmarkDetector initialized with backend='{self.backend}'")

    # ------------------------------------------------------------------
    def _init_mediapipe(self) -> None:
        try:
            import mediapipe as mp

            self._mp = mp
            self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
            )
            self.backend = "mediapipe"
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"MediaPipe unavailable ({exc}); will try Haar cascade fallback.")
            self._mp_face_mesh = None

    def _init_haar(self) -> None:
        try:
            face_path, eye_path = self._resolve_haar_cascade_paths()
            self._haar_face = cv2.CascadeClassifier(face_path)
            self._haar_eye = cv2.CascadeClassifier(eye_path)
            if self._haar_face.empty() or self._haar_eye.empty():
                raise RuntimeError(f"Haar cascade XML files failed to load ({face_path}, {eye_path}).")
            self.backend = "haar"
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Haar cascade fallback also failed: {exc}")

    @staticmethod
    def _resolve_haar_cascade_paths() -> Tuple[str, str]:
        """Prefer the cascades bundled in `assets/haarcascades/` - some
        opencv-python builds ship without the `cv2.data.haarcascades`
        directory populated, so relying on that alone is not reliable.
        Falls back to the OpenCV-provided path if the bundled copy is
        missing (e.g. a fresh checkout before assets were added).
        """
        bundled_dir = config.ASSETS_DIR / "haarcascades"
        bundled_face = bundled_dir / "haarcascade_frontalface_default.xml"
        bundled_eye = bundled_dir / "haarcascade_eye.xml"
        if bundled_face.exists() and bundled_eye.exists():
            return str(bundled_face), str(bundled_eye)
        return (
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
            cv2.data.haarcascades + "haarcascade_eye.xml",
        )

    # ------------------------------------------------------------------
    def process(self, frame_bgr: np.ndarray) -> FaceDetectionResult:
        if self.backend == "mediapipe":
            return self._process_mediapipe(frame_bgr)
        elif self.backend == "haar":
            return self._process_haar(frame_bgr)
        return FaceDetectionResult(success=False, backend=self.backend)

    # ------------------------------------------------------------------
    def _process_mediapipe(self, frame_bgr: np.ndarray) -> FaceDetectionResult:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._mp_face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return FaceDetectionResult(success=False, backend=self.backend)

        landmarks = results.multi_face_landmarks[0]
        pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks.landmark], dtype=np.float64)

        face_bbox = _bbox_from_points(pts, margin=0.15, w=w, h=h)

        left_eye_bbox = _bbox_from_points(pts[LEFT_EYE_OUTLINE_IDX], margin=0.6, w=w, h=h)
        right_eye_bbox = _bbox_from_points(pts[RIGHT_EYE_OUTLINE_IDX], margin=0.6, w=w, h=h)
        mouth_idx = [MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT]
        mouth_bbox = _bbox_from_points(pts[mouth_idx], margin=0.5, w=w, h=h)

        left_eye_crop = crop_region(frame_bgr, left_eye_bbox, config.EYE_IMG_SIZE)
        right_eye_crop = crop_region(frame_bgr, right_eye_bbox, config.EYE_IMG_SIZE)

        ear_left = _compute_ear(pts, LEFT_EYE_EAR_IDX)
        ear_right = _compute_ear(pts, RIGHT_EYE_EAR_IDX)
        ear = (ear_left + ear_right) / 2.0
        mar = _compute_mar(pts)

        pitch, yaw, roll = self._estimate_head_pose(pts, w, h)

        return FaceDetectionResult(
            success=True,
            backend=self.backend,
            face_bbox=face_bbox,
            left_eye_crop=left_eye_crop,
            right_eye_crop=right_eye_crop,
            left_eye_bbox=left_eye_bbox,
            right_eye_bbox=right_eye_bbox,
            mouth_bbox=mouth_bbox,
            ear=ear,
            mar=mar,
            head_pitch_deg=pitch,
            head_yaw_deg=yaw,
            head_roll_deg=roll,
            landmarks_px=pts,
        )

    def _estimate_head_pose(self, pts: np.ndarray, w: int, h: int):
        try:
            image_points = np.array(
                [pts[HEAD_POSE_IDX[k]] for k in (
                    "nose_tip", "chin", "left_eye_corner",
                    "right_eye_corner", "left_mouth_corner", "right_mouth_corner",
                )],
                dtype=np.float64,
            )
            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
                dtype=np.float64,
            )
            dist_coeffs = np.zeros((4, 1))
            success, rotation_vec, _ = cv2.solvePnP(
                _MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success:
                return None, None, None
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            pitch, yaw, roll = self._rotation_matrix_to_euler(rotation_mat)
            return pitch, yaw, roll
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Head pose estimation failed: {exc}")
            return None, None, None

    @staticmethod
    def _rotation_matrix_to_euler(R: np.ndarray):
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
            roll = math.atan2(R[2, 1], R[2, 2])
        else:
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0
            roll = math.atan2(-R[1, 2], R[1, 1])
        to_deg = lambda r: float(np.degrees(r))
        return to_deg(pitch), to_deg(yaw), to_deg(roll)

    # ------------------------------------------------------------------
    def _process_haar(self, frame_bgr: np.ndarray) -> FaceDetectionResult:
        """Degraded-but-functional fallback: bounding boxes only.

        EAR is approximated from the detected eye box's height/width ratio
        (a coarse proxy - real EAR needs eyelid landmarks). MAR and head
        pose are not available without landmarks and are left as None; the
        drowsiness logic treats None gracefully (see drowsiness_logic.py).
        """
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._haar_face.detectMultiScale(
            gray, scaleFactor=config.HAAR_SCALE_FACTOR, minNeighbors=config.HAAR_MIN_NEIGHBORS,
            minSize=(80, 80),
        )
        if len(faces) == 0:
            return FaceDetectionResult(success=False, backend=self.backend)

        fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        face_bbox = (fx, fy, fw, fh)
        face_roi_gray = gray[fy : fy + fh, fx : fx + fw]

        eyes = self._haar_eye.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=6, minSize=(20, 15))
        eyes = sorted(eyes, key=lambda b: b[0])[:2]  # left-to-right, at most 2

        left_eye_crop = right_eye_crop = None
        left_eye_bbox = right_eye_bbox = None
        ear_estimates = []

        for i, (ex, ey, ew, eh) in enumerate(eyes):
            abs_bbox = (fx + ex, fy + ey, ew, eh)
            crop = crop_region(frame_bgr, abs_bbox, config.EYE_IMG_SIZE)
            ear_estimates.append(eh / ew if ew > 0 else 0.0)
            if i == 0:
                left_eye_bbox, left_eye_crop = abs_bbox, crop
            else:
                right_eye_bbox, right_eye_crop = abs_bbox, crop

        # Normalize the crude height/width proxy into a range comparable to
        # true EAR (~0.15 closed .. ~0.35 open) via an empirical scale factor.
        ear = float(np.mean(ear_estimates)) * 0.5 if ear_estimates else None

        return FaceDetectionResult(
            success=True,
            backend=self.backend,
            face_bbox=face_bbox,
            left_eye_crop=left_eye_crop,
            right_eye_crop=right_eye_crop,
            left_eye_bbox=left_eye_bbox,
            right_eye_bbox=right_eye_bbox,
            mouth_bbox=None,
            ear=ear,
            mar=None,
            head_pitch_deg=None,
            head_yaw_deg=None,
            head_roll_deg=None,
            landmarks_px=None,
        )

    def close(self) -> None:
        if self._mp_face_mesh is not None:
            self._mp_face_mesh.close()
