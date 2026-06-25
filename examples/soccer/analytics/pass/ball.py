"""Ball detection helpers for pass analytics (reuses the tutorial YOLO ball model)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import supervision as sv
from ultralytics import YOLO

from analytics.class_ids import ROLE_BALL
from analytics.pass.possession import feet_xy

_SOCCER_DIR = Path(__file__).resolve().parent.parent.parent
BALL_MODEL_PATH = str(_SOCCER_DIR / "data" / "football-ball-detection.pt")


def _ball_centers(balls: sv.Detections) -> np.ndarray:
    """Ground positions (cx, y2) for each ball detection."""
    cx = (balls.xyxy[:, 0] + balls.xyxy[:, 2]) / 2
    cy = balls.xyxy[:, 3]
    return np.column_stack([cx, cy])


def select_best_ball(
    balls: sv.Detections,
    *,
    players: sv.Detections | None = None,
    prev_ball_xy: np.ndarray | None = None,
) -> sv.Detections:
    """Pick one ball box: nearest players, else previous position, else highest confidence."""
    if len(balls) <= 1:
        return balls
    if players is not None and len(players):
        feet = feet_xy(players)
        if len(feet):
            centers = _ball_centers(balls)
            dists = np.linalg.norm(
                centers[:, None, :] - feet[None, :, :], axis=2
            ).min(axis=1)
            pick = int(np.argmin(dists))
            return balls[pick : pick + 1]
    if prev_ball_xy is not None:
        centers = _ball_centers(balls)
        dists = np.linalg.norm(centers - prev_ball_xy, axis=1)
        pick = int(np.argmin(dists))
        return balls[pick : pick + 1]
    if balls.confidence is None:
        return balls[:1]
    pick = int(np.argmax(balls.confidence))
    return balls[pick : pick + 1]


def _ball_row_for_merge(ball: sv.Detections, players: sv.Detections) -> sv.Detections:
    """Make a single ball detection compatible with tracked player rows for merge."""
    if len(ball) == 0:
        return ball
    ball = ball[:1]
    kwargs: dict = {
        "xyxy": ball.xyxy,
        "class_id": ball.class_id,
        "confidence": ball.confidence,
    }
    if players.tracker_id is not None:
        kwargs["tracker_id"] = np.array([-1], dtype=int)
    if players.data:
        ball_data: dict = {}
        n_players = len(players)
        for key, val in players.data.items():
            arr = np.asarray(val)
            if arr.ndim == 0:
                ball_data[key] = arr
            elif len(arr) == n_players:
                if np.issubdtype(arr.dtype, np.floating):
                    ball_data[key] = np.array([np.nan], dtype=arr.dtype)
                else:
                    ball_data[key] = np.array([-1], dtype=arr.dtype)
            else:
                ball_data[key] = arr[:1]
        kwargs["data"] = ball_data
    return sv.Detections(**kwargs)


def attach_ball(
    dets: sv.Detections,
    ball_dets: sv.Detections,
    *,
    prev_ball_xy: np.ndarray | None = None,
) -> sv.Detections:
    """Append the best ball row to a player/GK detections frame."""
    ball = select_best_ball(ball_dets, players=dets, prev_ball_xy=prev_ball_xy)
    if len(ball) == 0:
        return dets
    ball = _ball_row_for_merge(ball, dets)
    return sv.Detections.merge([dets, ball])


def create_ball_detector(
    *,
    device: str = "cpu",
    model_path: str | None = None,
    threshold: float = 0.2,
) -> Callable[[np.ndarray], sv.Detections]:
    """Return ``detect(frame_bgr) -> sv.Detections`` for the football ball YOLO model."""
    path = model_path or BALL_MODEL_PATH
    model = YOLO(str(path)).to(device=device)

    def _detect(frame: np.ndarray) -> sv.Detections:
        results = model.predict(frame, conf=threshold, verbose=False, device=device)[0]
        dets = sv.Detections.from_ultralytics(results)
        if len(dets) == 0:
            return dets
        return sv.Detections(
            xyxy=dets.xyxy,
            confidence=dets.confidence,
            class_id=np.full(len(dets), ROLE_BALL, dtype=int),
        )

    return _detect
