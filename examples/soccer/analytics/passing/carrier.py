"""Ball motion helpers for pass analytics.

Per-frame team/carrier possession timelines are not implemented here yet;
pass detection keeps its state machine in ``passes.scan_possession_events``.
Revisit this module if we add team possession % or time-on-ball stats.
"""

from __future__ import annotations

import numpy as np

from .pitch_helpers import image_to_pitch_m


class BallPositionHistory:
    """Recent ball ground positions for speed when picking freeze frames."""

    def __init__(self, *, max_samples: int = 12) -> None:
        self._max_samples = max_samples
        self._samples: list[tuple[int, np.ndarray]] = []

    def record(self, frame_idx: int, ball: np.ndarray | None) -> None:
        if ball is None:
            return
        self._samples.append((int(frame_idx), np.asarray(ball, dtype=np.float64)))
        if len(self._samples) > self._max_samples:
            self._samples = self._samples[-self._max_samples :]

    def speed(
        self,
        frame_idx: int,
        *,
        lookback_frames: int,
        fps: float,
        transformer=None,
    ) -> float | None:
        """Ball speed over the lookback window (m/s with homography, else px/s)."""
        if fps <= 0 or lookback_frames < 1:
            return None
        window = [
            (f, p)
            for f, p in self._samples
            if frame_idx - lookback_frames <= f <= frame_idx
        ]
        if len(window) < 2:
            return None
        f0, p0 = window[0]
        f1, p1 = window[-1]
        if f1 <= f0:
            return None
        dt = (f1 - f0) / fps
        if dt <= 0:
            return None

        if transformer is not None:
            pts = np.stack([p0, p1], axis=0).astype(np.float32)
            pitch = image_to_pitch_m(pts, transformer)
            if pitch is None:
                return None
            dist = float(np.linalg.norm(pitch[1] - pitch[0]))
        else:
            dist = float(np.linalg.norm(p1 - p0))
        return dist / dt
