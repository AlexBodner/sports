"""Pass detection and ball possession analytics."""

from __future__ import annotations

from .ball import attach_ball, create_ball_detector
from .carrier import BallPositionHistory, TrackPositionHistory
from .pass_options import (
    PassOption,
    PassWeights,
    remap_lane_debug_to_pitch_cm,
    top_pass_options,
)
from .passes import (
    InferredPass,
    InferredTurnover,
    PassDetectionConfig,
    PassQualityScorer,
    PossessionScanResult,
    passes_for_overlay,
    scan_possession_events,
)
from .pitch_helpers import (
    image_to_pitch_cm,
    image_to_pitch_m,
    lane_scoring_transformer_for_frame,
    pitch_attack_direction,
)
from .possession import (
    Carrier,
    ball_xy,
    bbox_center_xy,
    carrier_from_tracker_id,
    find_control_carrier,
)

__all__ = [
    "BallPositionHistory",
    "Carrier",
    "InferredPass",
    "InferredTurnover",
    "PassDetectionConfig",
    "PassOption",
    "PassQualityScorer",
    "PassWeights",
    "PossessionScanResult",
    "TrackPositionHistory",
    "attach_ball",
    "ball_xy",
    "bbox_center_xy",
    "carrier_from_tracker_id",
    "create_ball_detector",
    "find_control_carrier",
    "image_to_pitch_cm",
    "image_to_pitch_m",
    "lane_scoring_transformer_for_frame",
    "passes_for_overlay",
    "pitch_attack_direction",
    "remap_lane_debug_to_pitch_cm",
    "scan_possession_events",
    "top_pass_options",
]
