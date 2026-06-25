"""Pass alternatives: freeze-moment planning and the PASS_ALTERNATIVES runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import supervision as sv

from analytics.annotations import (
    annotate_ball,
    annotate_players,
    draw_branding_tag,
    draw_carrier_halo,
    draw_hud_bar,
    draw_pass_overlay,
    draw_pitch_keypoints_debug,
    draw_radar_minimap,
)
from analytics.carrier import BallPositionHistory, TrackPositionHistory
from analytics.clip_pipeline import ClipAnalysis, compute_clip_analysis
from analytics.homography import (
    image_to_pitch_cm,
    image_to_pitch_m,
    lane_scoring_transformer_for_frame,
    pitch_attack_direction,
)
from analytics.pass_options import PassOption, PassWeights, top_pass_options
from analytics.player_motion import JoystickDotSmoother, carrier_kalman_direction, open_video
from analytics.possession import (
    Carrier,
    ball_xy,
    bbox_center_xy,
    feet_xy,
    find_control_carrier,
    player_mask,
)


@dataclass(frozen=True)
class PassEvent:
    frame_idx: int
    carrier: Carrier
    options: list[PassOption]
    top_score: float

MAX_RAMP_HOLD = 6
DEFAULT_SLOWDOWN_RAMP_SECONDS = 0.72
DEFAULT_OPTION_REVEAL_SECONDS = 0.6
DEFAULT_FREEZE_SECONDS = 2.5
DEFAULT_FINAL_OPTION_EXTRA_SECONDS = 1.0


def _score_options(
    dets: sv.Detections,
    carrier: Carrier,
    *,
    weights: PassWeights,
    transformer: Any | None,
    metric: bool,
) -> list[PassOption]:
    motion_dir = None
    if weights.use_carrier_motion:
        motion_dir = carrier_kalman_direction(
            dets,
            carrier.index,
            transformer=transformer if metric else None,
        )

    if metric and transformer is not None:
        feet_img = feet_xy(dets)
        pitch_feet = image_to_pitch_m(feet_img, transformer)
        pitch_cm = image_to_pitch_cm(feet_img, transformer)
        body_pitch_m = image_to_pitch_m(bbox_center_xy(dets), transformer)
        if pitch_feet is not None and pitch_cm is not None:
            attack_dir = pitch_attack_direction(
                dets,
                carrier.team,
                transformer,
                player_mask_fn=player_mask,
                feet_fn=feet_xy,
            )
            return top_pass_options(
                dets,
                carrier,
                k=3,
                weights=weights,
                attack_dir=attack_dir,
                positions=pitch_feet,
                carrier_motion_dir=motion_dir,
                pitch_cm=pitch_cm,
                body_pitch_m=body_pitch_m,
            )
    return top_pass_options(
        dets,
        carrier,
        k=3,
        weights=weights,
        carrier_motion_dir=motion_dir,
    )


def _freeze_moment_score(
    pass_top: float,
    carrier: Carrier,
    *,
    ball_speed: float | None,
    weights: PassWeights,
    metric: bool,
) -> float | None:
    if weights.use_ball_control_gate and ball_speed is not None:
        if metric:
            if ball_speed >= weights.ball_speed_skip_m:
                return None
            ref, cap = weights.ball_speed_ref_m, weights.ball_speed_max_m
            tight_ref = weights.carrier_tight_ref_m
        else:
            if ball_speed >= weights.ball_speed_skip_px_s:
                return None
            ref, cap = weights.ball_speed_ref_px_s, weights.ball_speed_max_px_s
            tight_ref = weights.carrier_tight_ref_px

        score = pass_top
        if ball_speed <= ref:
            score += 0.05
        elif ball_speed < cap:
            t = (ball_speed - ref) / (cap - ref)
            score -= weights.ball_speed_penalty * t
        else:
            score -= weights.ball_speed_penalty

        if carrier.distance < tight_ref:
            score += weights.carrier_tight_bonus * (
                1.0 - carrier.distance / tight_ref
            )
        return score

    score = pass_top
    tight_ref = (
        weights.carrier_tight_ref_m if metric else weights.carrier_tight_ref_px
    )
    if carrier.distance < tight_ref:
        score += weights.carrier_tight_bonus * (1.0 - carrier.distance / tight_ref)
    return score


def _resolve_freeze_frame_earlier(
    event: PassEvent,
    by_frame: dict[int, PassEvent],
    *,
    weights: PassWeights,
    metric: bool,
    instant_speed_by_frame: dict[int, float],
) -> PassEvent:
    if not weights.freeze_nudge_earlier:
        return event
    slack = weights.freeze_nudge_score_slack
    eps = (
        weights.freeze_separation_eps_m
        if metric
        else weights.freeze_separation_eps_px
    )
    best = event

    while True:
        prev = by_frame.get(best.frame_idx - 1)
        if prev is None or prev.top_score < best.top_score - slack:
            break
        separating = best.carrier.distance > prev.carrier.distance + eps
        instant = instant_speed_by_frame.get(best.frame_idx)
        fast_ball = instant is not None and (
            (metric and instant >= weights.freeze_release_ball_speed_skip_m)
            or (not metric and instant >= weights.freeze_release_ball_speed_skip_px_s)
        )
        if not (separating or fast_ball):
            break
        best = prev

    prev = by_frame.get(best.frame_idx - 1)
    if (
        prev is not None
        and best.top_score > prev.top_score
        and prev.top_score >= best.top_score - slack
    ):
        best = prev

    return best


def _apply_freeze_frame_nudges(
    candidates: list[PassEvent],
    *,
    weights: PassWeights,
    metric: bool,
    instant_speed_by_frame: dict[int, float],
) -> list[PassEvent]:
    by_frame = {e.frame_idx: e for e in candidates}
    resolved: dict[int, PassEvent] = {}
    for event in candidates:
        nudged = _resolve_freeze_frame_earlier(
            event,
            by_frame,
            weights=weights,
            metric=metric,
            instant_speed_by_frame=instant_speed_by_frame,
        )
        prev = resolved.get(nudged.frame_idx)
        if prev is None or nudged.top_score > prev.top_score:
            resolved[nudged.frame_idx] = nudged
    return sorted(resolved.values(), key=lambda e: e.frame_idx)


def _select_pass_moments(
    candidates: list[PassEvent],
    *,
    weights: PassWeights,
    min_gap_frames: int,
    max_events: int | None,
) -> list[PassEvent]:
    if not candidates:
        return []

    by_frame = sorted(candidates, key=lambda e: e.frame_idx)
    score_at = {e.frame_idx: e.top_score for e in by_frame}
    half = weights.freeze_local_peak_half_window

    peaks: list[PassEvent] = []
    for event in by_frame:
        if event.top_score < weights.freeze_min_pick_score:
            continue
        if event.options[0].score < weights.freeze_min_pass_score:
            continue
        if weights.freeze_detect_local_peaks:
            f = event.frame_idx
            neighbor_scores = [
                score_at.get(f + d, -1.0)
                for d in range(-half, half + 1)
                if d != 0 and (f + d) in score_at
            ]
            if neighbor_scores and event.top_score <= max(neighbor_scores):
                continue
        peaks.append(event)

    peaks.sort(key=lambda e: e.top_score, reverse=True)
    chosen: list[PassEvent] = []
    for event in peaks:
        if all(abs(event.frame_idx - c.frame_idx) >= min_gap_frames for c in chosen):
            chosen.append(event)
        if max_events is not None and len(chosen) >= max_events:
            break
    chosen.sort(key=lambda e: e.frame_idx)
    return chosen


def _slowdown_hold_count(
    frames_until_event: int,
    *,
    ramp_frames: int,
    max_extra_holds: int = MAX_RAMP_HOLD,
) -> int:
    """Repeat count for live frames as playback eases into a freeze."""
    if frames_until_event <= 0 or frames_until_event > ramp_frames:
        return 1
    t = 1.0 - frames_until_event / ramp_frames
    return 1 + int((t * t) * max(0, max_extra_holds - 1))


def _annotate_live(
    frame: np.ndarray,
    dets: sv.Detections,
    *,
    keypoints: Any | None = None,
    pitch_confidence: float = 0.9,
    metric: bool = False,
    show_radar: bool = True,
    radar_transformer: Any | None = None,
    locked_goal_defenders: tuple[int, int] | None = None,
    debug_pitch_keypoints: bool = False,
    show_kalman_joystick: bool = False,
    dot_smoother: JoystickDotSmoother | None = None,
) -> np.ndarray:
    """Live (non-freeze) frame: players + ball + carrier halo + radar minimap."""
    frame = annotate_players(
        frame,
        dets,
        show_kalman_joystick=show_kalman_joystick,
        dot_smoother=dot_smoother,
        show_tracker_ids=True,
    )
    frame = annotate_ball(frame, dets)
    carrier = find_control_carrier(
        dets,
        transformer=radar_transformer if metric else None,
    )
    if carrier is not None:
        feet = feet_xy(dets)[carrier.index]
        draw_carrier_halo(frame, (int(feet[0]), int(feet[1])))
    if show_radar and metric and keypoints is not None:
        frame = draw_radar_minimap(
            frame,
            dets,
            keypoints,
            pitch_confidence=pitch_confidence,
            locked_goal_defenders=locked_goal_defenders,
            debug_keypoints=True,
        )
    if debug_pitch_keypoints and keypoints is not None:
        frame = draw_pitch_keypoints_debug(
            frame, keypoints, confidence_threshold=pitch_confidence
        )
    frame = draw_hud_bar(frame, "PASS ALTERNATIVES")
    return draw_branding_tag(frame)


def plan_pass_events(
    pass_frames: list[tuple[int, sv.Detections]],
    *,
    fps: float,
    frame_transforms: dict[int, Any | None],
    keypoints_by_frame: dict[int, Any | None],
    weights: PassWeights | None = None,
    max_events: int | None = None,
    min_gap_frames: int = 90,
    pitch_confidence: float = 0.9,
) -> list[PassEvent]:
    """Detect cinematic pass freeze moments from goal-distance pass frames."""
    weights = weights or PassWeights.metric()
    transformers = frame_transforms
    max_px = weights.freeze_carrier_max_distance_px
    max_m = weights.freeze_carrier_max_distance_m
    require_both = weights.freeze_require_both_spaces

    history = TrackPositionHistory()
    ball_history = BallPositionHistory()
    candidates: list[PassEvent] = []
    instant_speed_by_frame: dict[int, float] = {}

    for frame_idx, dets in pass_frames:
        kps = keypoints_by_frame.get(frame_idx) if keypoints_by_frame else None
        transformer = lane_scoring_transformer_for_frame(
            transformers, frame_idx, kps, pitch_confidence=pitch_confidence
        )
        ball_history.record(frame_idx, ball_xy(dets))
        feet_img = feet_xy(dets)
        hist_xy = feet_img
        if transformer is not None:
            pitch_feet = image_to_pitch_m(feet_img, transformer)
            if pitch_feet is not None:
                hist_xy = pitch_feet
        history.record_frame(frame_idx, dets, hist_xy)

        carrier = find_control_carrier(
            dets,
            max_distance_px=max_px,
            transformer=transformer,
            max_distance_m=max_m,
            require_both_spaces=require_both and transformer is not None,
        )
        if carrier is None or frame_idx < 30:
            continue
        options = _score_options(
            dets, carrier, weights=weights, transformer=transformer, metric=True
        )
        if len(options) < 2:
            continue
        top = options[0]
        if top.length < weights.min_length or top.length > weights.max_length:
            continue
        ball_speed = ball_history.speed(
            frame_idx,
            lookback_frames=weights.ball_speed_lookback_frames,
            fps=fps,
            transformer=transformer,
        )
        pick_score = _freeze_moment_score(
            top.score, carrier, ball_speed=ball_speed, weights=weights, metric=True
        )
        if pick_score is None:
            continue
        instant = ball_history.speed(
            frame_idx, lookback_frames=1, fps=fps, transformer=transformer
        )
        if instant is not None:
            instant_speed_by_frame[frame_idx] = instant
        candidates.append(PassEvent(frame_idx, carrier, options, pick_score))

    candidates = _apply_freeze_frame_nudges(
        candidates,
        weights=weights,
        metric=True,
        instant_speed_by_frame=instant_speed_by_frame,
    )
    return _select_pass_moments(
        candidates,
        weights=weights,
        min_gap_frames=min_gap_frames,
        max_events=max_events,
    )


def _render_pass_alternatives(args, analysis: ClipAnalysis) -> None:
    """Render PASS_ALTERNATIVES using a shared :class:`ClipAnalysis`."""
    metric = analysis.metric
    locks = analysis.locks("goal_distance")
    locked_goals = locks.locked_goal_defenders
    pass_by_frame = analysis.pass_by_frame
    pitch_confidence = 0.9
    weights = PassWeights.metric()
    freeze_events = analysis.pass_alternative_events
    events_by_frame = {e.frame_idx: e for e in freeze_events}
    event_frames = sorted(events_by_frame)
    fps = float(analysis.fps)
    ramp_frames = max(1, int(round(DEFAULT_SLOWDOWN_RAMP_SECONDS * fps)))
    reveal_frames = max(4, int(round(DEFAULT_OPTION_REVEAL_SECONDS * fps)))
    dot_smoother = JoystickDotSmoother()
    debug_pitch_keypoints = getattr(args, "debug_pitch_keypoints", False)

    cap, _, width, height = open_video(args.source_video_path)
    with sv.VideoSink(args.target_video_path, sv.VideoInfo(width, height, fps)) as sink:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if analysis.max_frames is not None and frame_idx > analysis.max_frames:
                break
            dets = pass_by_frame.get(frame_idx)
            if dets is None:
                continue

            kps = metric.keypoints.get(frame_idx)
            transformer = lane_scoring_transformer_for_frame(
                metric.transforms, frame_idx, kps, pitch_confidence=pitch_confidence
            )
            live = _annotate_live(
                frame,
                dets,
                keypoints=kps,
                pitch_confidence=pitch_confidence,
                metric=True,
                radar_transformer=transformer,
                locked_goal_defenders=locked_goals,
                debug_pitch_keypoints=debug_pitch_keypoints,
                show_kalman_joystick=True,
                dot_smoother=dot_smoother,
            )
            frames_until = next(
                (ef - frame_idx for ef in event_frames if ef >= frame_idx), None
            )
            hold = (
                _slowdown_hold_count(frames_until, ramp_frames=ramp_frames)
                if frames_until is not None
                else 1
            )
            for _ in range(hold):
                sink.write_frame(live)

            if frame_idx in events_by_frame:
                event = events_by_frame[frame_idx]
                n_options = min(3, len(event.options))
                overlay_kwargs = dict(
                    weights=weights,
                    metric=True,
                    keypoints=kps,
                    pitch_confidence=pitch_confidence,
                    transformer=transformer,
                    show_lane_debug=kps is not None,
                    show_radar=True,
                    locked_goal_defenders=locked_goals,
                    debug_pitch_keypoints=debug_pitch_keypoints,
                )
                phases: list[tuple[int, int]] = [(0, reveal_frames)]
                phases.extend((i, reveal_frames) for i in range(1, n_options + 1))
                min_freeze = sum(h for _, h in phases)
                extra_hold = max(0, int(round(DEFAULT_FREEZE_SECONDS * fps)) - min_freeze)
                final_extra = max(
                    4, int(round(DEFAULT_FINAL_OPTION_EXTRA_SECONDS * fps))
                )
                if phases:
                    phases[-1] = (
                        phases[-1][0],
                        phases[-1][1] + extra_hold + final_extra,
                    )
                for revealed, phase_hold in phases:
                    for step in range(phase_hold):
                        progress = (step + 1) / max(phase_hold, 1)
                        overlay = draw_pass_overlay(
                            frame,
                            dets,
                            event,
                            revealed_options=revealed,
                            reveal_progress=progress,
                            show_kalman_joystick=True,
                            dot_smoother=dot_smoother,
                            **overlay_kwargs,
                        )
                        sink.write_frame(overlay)
    cap.release()
    print(f"Wrote {args.target_video_path}")


def run_pass_alternatives(args, analysis: ClipAnalysis | None = None) -> None:
    """Render cinematic pass-alternative freeze frames to ``args.target_video_path``."""
    if analysis is None:
        analysis = compute_clip_analysis(args, need_homography=True)
    _render_pass_alternatives(args, analysis)
