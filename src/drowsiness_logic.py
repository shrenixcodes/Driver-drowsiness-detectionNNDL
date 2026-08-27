"""
Temporal decision logic: turns a raw per-frame drowsiness probability (from
either deep model) plus classical EAR/MAR signals into a stable, debounced
system state.

This module is what prevents a single bad frame (a blink, a motion blur
frame, a brief misdetection) from triggering a false alarm:

    raw probability -> rolling-average smoothing -> threshold ->
    consecutive high-risk frame counter -> warning

All thresholds/window sizes are read from `src.config` by default but can be
overridden per instance (the Streamlit sidebar does this).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from src import config

ALERT, DROWSY, HIGH_RISK = "ALERT", "DROWSY", "HIGH_RISK"


@dataclass
class DrowsinessState:
    status: str
    smoothed_probability: float
    raw_probability: float
    eye_state: str  # "OPEN" | "CLOSED" | "UNKNOWN"
    perclos: float  # fraction of recent frames with eyes closed
    blink_rate_per_min: float
    total_blinks: int
    yawn_in_progress: bool
    total_yawns: int
    consecutive_closed_frames: int
    warning_active: bool
    high_risk_streak: int


def heuristic_probability_from_signals(
    ear: Optional[float],
    consecutive_closed_frames: int,
    ear_threshold: float = config.EAR_THRESHOLD,
    prolonged_frames: int = config.EYE_CLOSED_CONSEC_FRAMES,
) -> float:
    """Classical-CV fallback probability estimate, used when no trained deep
    model is loaded ("Heuristic mode"). Purely a function of EAR and how long
    the eyes have been continuously closed - no learned weights involved.
    """
    if ear is None:
        return 0.0
    if ear >= ear_threshold:
        return 0.05
    # Below threshold: base probability scales with how far below threshold,
    # then grows further the longer the closure persists (captures
    # "prolonged eye closure" being worse than a blink).
    closure_ratio = max(0.0, min(1.0, (ear_threshold - ear) / ear_threshold))
    base = 0.4 + 0.4 * closure_ratio
    duration_boost = min(0.4, 0.4 * (consecutive_closed_frames / max(1, prolonged_frames)))
    return float(min(1.0, base + duration_boost))


class DrowsinessLogic:
    def __init__(
        self,
        drowsy_threshold: float = config.DROWSY_THRESHOLD,
        high_risk_threshold: float = config.HIGH_RISK_THRESHOLD,
        smoothing_window: int = config.SMOOTHING_WINDOW,
        warning_duration: int = config.WARNING_DURATION,
        ear_threshold: float = config.EAR_THRESHOLD,
        mar_threshold: float = config.MAR_YAWN_THRESHOLD,
        blink_min_consec_frames: int = config.BLINK_MIN_CONSEC_FRAMES,
        prolonged_closure_frames: int = config.EYE_CLOSED_CONSEC_FRAMES,
    ):
        self.drowsy_threshold = drowsy_threshold
        self.high_risk_threshold = high_risk_threshold
        self.warning_duration = warning_duration
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.blink_min_consec_frames = blink_min_consec_frames
        self.prolonged_closure_frames = prolonged_closure_frames

        self._prob_window = deque(maxlen=smoothing_window)
        self._perclos_window = deque(maxlen=max(smoothing_window * 3, 30))
        self._blink_timestamps = deque()
        self._start_time = time.time()

        self._consecutive_closed_frames = 0
        self._currently_in_blink = False
        self._high_risk_streak = 0
        self._recovery_streak = 0
        self._warning_active = False
        self._total_blinks = 0

        self._mar_above_streak = 0
        self._yawn_in_progress = False
        self._total_yawns = 0
        self._yawn_cooldown = 0

    @property
    def consecutive_closed_frames(self) -> int:
        return self._consecutive_closed_frames

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.__init__(
            self.drowsy_threshold, self.high_risk_threshold, self._prob_window.maxlen,
            self.warning_duration, self.ear_threshold, self.mar_threshold,
            self.blink_min_consec_frames, self.prolonged_closure_frames,
        )

    # ------------------------------------------------------------------
    def _update_eye_signals(self, ear: Optional[float]) -> str:
        if ear is None:
            self._perclos_window.append(0)
            return "UNKNOWN"

        eye_closed = ear < self.ear_threshold
        self._perclos_window.append(1 if eye_closed else 0)

        if eye_closed:
            self._consecutive_closed_frames += 1
            self._currently_in_blink = True
        else:
            if self._currently_in_blink and self._consecutive_closed_frames >= self.blink_min_consec_frames:
                self._total_blinks += 1
                self._blink_timestamps.append(time.time())
            self._currently_in_blink = False
            self._consecutive_closed_frames = 0

        # Prune blink timestamps older than 60s for a rolling blink-rate/min
        cutoff = time.time() - 60
        while self._blink_timestamps and self._blink_timestamps[0] < cutoff:
            self._blink_timestamps.popleft()

        return "CLOSED" if eye_closed else "OPEN"

    def _update_yawn_signal(self, mar: Optional[float]) -> None:
        if mar is None:
            self._mar_above_streak = 0
            self._yawn_in_progress = False
            return

        if mar >= self.mar_threshold:
            self._mar_above_streak += 1
        else:
            self._mar_above_streak = 0
            self._yawn_in_progress = False

        if self._mar_above_streak >= 8 and self._yawn_cooldown == 0:
            self._yawn_in_progress = True
            self._total_yawns += 1
            self._yawn_cooldown = 45  # frames before another yawn can be counted

        if self._yawn_cooldown > 0:
            self._yawn_cooldown -= 1

    def _blink_rate_per_min(self) -> float:
        elapsed = max(1.0, time.time() - self._start_time)
        window = min(60.0, elapsed)
        if window <= 0:
            return 0.0
        return len(self._blink_timestamps) * (60.0 / window)

    # ------------------------------------------------------------------
    def update(self, raw_probability: float, ear: Optional[float] = None, mar: Optional[float] = None) -> DrowsinessState:
        raw_probability = float(max(0.0, min(1.0, raw_probability)))
        eye_state = self._update_eye_signals(ear)
        self._update_yawn_signal(mar)

        self._prob_window.append(raw_probability)
        smoothed = sum(self._prob_window) / len(self._prob_window)

        perclos = sum(self._perclos_window) / len(self._perclos_window) if self._perclos_window else 0.0

        # Base status from the smoothed learned/heuristic probability
        if smoothed >= self.high_risk_threshold:
            status = HIGH_RISK
        elif smoothed >= self.drowsy_threshold:
            status = DROWSY
        else:
            status = ALERT

        # Safety override: sustained eye closure escalates status even if the
        # model's smoothed probability hasn't caught up yet (microsleep guard).
        if self._consecutive_closed_frames >= self.prolonged_closure_frames * 2:
            status = HIGH_RISK
        elif self._consecutive_closed_frames >= self.prolonged_closure_frames:
            status = HIGH_RISK if status == HIGH_RISK else DROWSY

        # Debounced alarm: only sound after `warning_duration` consecutive
        # HIGH_RISK (smoothed) frames, and keep it on with hysteresis so it
        # doesn't chatter on/off around the boundary.
        if status == HIGH_RISK:
            self._high_risk_streak += 1
            self._recovery_streak = 0
        else:
            self._high_risk_streak = 0
            if smoothed < self.drowsy_threshold:
                self._recovery_streak += 1
            else:
                self._recovery_streak = 0

        if not self._warning_active and self._high_risk_streak >= self.warning_duration:
            self._warning_active = True
        elif self._warning_active and self._recovery_streak >= 5:
            self._warning_active = False

        return DrowsinessState(
            status=status,
            smoothed_probability=smoothed,
            raw_probability=raw_probability,
            eye_state=eye_state,
            perclos=perclos,
            blink_rate_per_min=self._blink_rate_per_min(),
            total_blinks=self._total_blinks,
            yawn_in_progress=self._yawn_in_progress,
            total_yawns=self._total_yawns,
            consecutive_closed_frames=self._consecutive_closed_frames,
            warning_active=self._warning_active,
            high_risk_streak=self._high_risk_streak,
        )
