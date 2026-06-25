"""Import bridge for ``analytics.pass`` (``pass`` is a reserved Python keyword).

External modules must import pass analytics through this module or via relative
imports inside ``analytics/pass/``.
"""

from __future__ import annotations

import importlib

ball = importlib.import_module("analytics.pass.ball")
carrier = importlib.import_module("analytics.pass.carrier")
possession = importlib.import_module("analytics.pass.possession")
passes = importlib.import_module("analytics.pass.passes")
pass_options = importlib.import_module("analytics.pass.pass_options")
pitch_helpers = importlib.import_module("analytics.pass.pitch_helpers")

attach_ball = ball.attach_ball
create_ball_detector = ball.create_ball_detector

BallPositionHistory = carrier.BallPositionHistory
TrackPositionHistory = carrier.TrackPositionHistory
Carrier = possession.Carrier

ball_xy = possession.ball_xy
bbox_center_xy = possession.bbox_center_xy
carrier_from_tracker_id = possession.carrier_from_tracker_id
find_control_carrier = possession.find_control_carrier

from analytics.player_motion import feet_xy, player_mask  # noqa: E402

InferredPass = passes.InferredPass
InferredTurnover = passes.InferredTurnover
PassDetectionConfig = passes.PassDetectionConfig
PassQualityScorer = passes.PassQualityScorer
PossessionScanResult = passes.PossessionScanResult
passes_for_overlay = passes.passes_for_overlay
scan_possession_events = passes.scan_possession_events

PassOption = pass_options.PassOption
PassWeights = pass_options.PassWeights
remap_lane_debug_to_pitch_cm = pass_options.remap_lane_debug_to_pitch_cm
top_pass_options = pass_options.top_pass_options

image_to_pitch_cm = pitch_helpers.image_to_pitch_cm
image_to_pitch_m = pitch_helpers.image_to_pitch_m
lane_scoring_transformer_for_frame = pitch_helpers.lane_scoring_transformer_for_frame
pitch_attack_direction = pitch_helpers.pitch_attack_direction
