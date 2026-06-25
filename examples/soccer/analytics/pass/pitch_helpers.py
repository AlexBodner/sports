"""Pass analytics pitch-space helpers."""

from __future__ import annotations

import numpy as np
import supervision as sv
from sports.common.view import ViewTransformer

from analytics.goalkeepers import image_to_pitch_cm
from analytics.homography import homography_from_keypoints_radar

__all__ = [
    "image_to_pitch_cm",
    "image_to_pitch_m",
    "lane_scoring_transformer_for_frame",
    "pitch_attack_direction",
]


def image_to_pitch_m(
    points_xy: np.ndarray, transformer: ViewTransformer | None
) -> np.ndarray | None:
    cm = image_to_pitch_cm(points_xy, transformer)
    if cm is None:
        return None
    return cm / 100.0


def lane_scoring_transformer_for_frame(
    speed_transforms: dict[int, ViewTransformer | None] | None,
    frame_idx: int,
    keypoints: sv.KeyPoints | None,
    *,
    pitch_confidence: float = 0.9,
) -> ViewTransformer | None:
    """Prefer gated speed H; fall back to per-frame radar fit for lane scoring only."""
    if speed_transforms is not None:
        speed_t = speed_transforms.get(int(frame_idx))
        if speed_t is not None:
            return speed_t
    return homography_from_keypoints_radar(keypoints, confidence=pitch_confidence)


def pitch_attack_direction(
    detections: sv.Detections,
    carrier_team: int,
    transformer: ViewTransformer,
    *,
    player_mask_fn,
    feet_fn,
) -> np.ndarray:
    from analytics.class_ids import TEAM_LEFT
    from analytics.geometry import unit

    pmask = player_mask_fn(detections)
    if not pmask.any():
        return np.array([1.0, 0.0])

    feet = feet_fn(detections)[pmask]
    teams = detections.data["team"][pmask]
    pitch_xy = image_to_pitch_m(feet, transformer)
    if pitch_xy is None:
        return np.array([1.0, 0.0])

    own = pitch_xy[teams == carrier_team]
    opp = pitch_xy[teams == (1 - carrier_team)]
    if len(own) == 0 or len(opp) == 0:
        return np.array([1.0, 0.0]) if carrier_team == TEAM_LEFT else np.array([-1.0, 0.0])

    return unit(opp.mean(axis=0) - own.mean(axis=0))
