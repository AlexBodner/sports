"""OpenCV overlays for pass analytics and lane visualization."""

from __future__ import annotations

import colorsys
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from analytics.modes.pass_alternatives import PassEvent
    from analytics.player_motion import (
        JoystickDotSmoother,
        KalmanSpeedDisplaySmoother,
    )

import cv2
import numpy as np
import supervision as sv

from analytics.goalkeepers import image_to_pitch_cm
from analytics.homography import (
    HOMOGRAPHY_RANSAC_REPROJ_THRESH,
    PITCH_CONFIG,
    ViewTransformer,
    draw_pitch,
    draw_points_on_pitch,
    pitch_circle_to_image,
    pitch_cm_to_image,
    pitch_keypoint_accept_mask,
    pitch_keypoint_confidence,
    render_radar,
)
from analytics.pass_imports import image_to_pitch_m, pitch_attack_direction
from analytics.pass_imports import PassOption, ball_xy, feet_xy, player_mask
from analytics.class_ids import ROLE_GOALKEEPER, ROLE_PLAYER
from analytics.draw_helpers import cv2_safe_text, draw_text_shadow
from analytics.player_motion import (
    _dot_radius_for_ellipse,
    _joystick_dot_reach,
    draw_speed_badge,
    kalman_speed_stick,
    player_ellipse_geometry,
)

ROBOFLOW_PURPLE = sv.Color.from_hex("#8315F9")
ROBOFLOW_PURPLE_BGR = ROBOFLOW_PURPLE.as_bgr()
TEAM_COLORS = [
    sv.Color.from_hex("#00BFFF"),
    sv.Color.from_hex("#FF1493"),
    sv.Color.from_hex("#FFD700"),
]
TEAM_PALETTE = sv.ColorPalette(TEAM_COLORS)
BALL_COLOR = sv.Color.from_hex("#FFD700")
KALMAN_FACING_BGR = (0, 200, 255)

_ELLIPSE = sv.EllipseAnnotator(
    color=TEAM_PALETTE, color_lookup=sv.ColorLookup.CLASS, thickness=2
)
_LABEL = sv.LabelAnnotator(
    text_position=sv.Position.BOTTOM_CENTER,
    text_scale=0.45,
    text_thickness=1,
    border_radius=4,
    color=TEAM_PALETTE,
    color_lookup=sv.ColorLookup.CLASS,
)
_BALL_TRI = sv.TriangleAnnotator(
    color=BALL_COLOR, base=14, height=18, color_lookup=sv.ColorLookup.INDEX
)
_GK_ELLIPSE = sv.EllipseAnnotator(
    color=sv.Color.from_hex("#E8E8E8"), thickness=2
)


def track_id_palette(*, size: int = 64) -> sv.ColorPalette:
    """Distinct hues for ``ColorLookup.TRACK`` (stable color per ``tracker_id``)."""
    colors: list[sv.Color] = []
    for i in range(size):
        hue = (i * 0.618033988749895) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        colors.append(
            sv.Color(
                r=int(red * 255),
                g=int(green * 255),
                b=int(blue * 255),
            )
        )
    return sv.ColorPalette(colors)


_TRACK_ID_PALETTE = track_id_palette()
_TRACK_ELLIPSE = sv.EllipseAnnotator(
    color=_TRACK_ID_PALETTE,
    color_lookup=sv.ColorLookup.TRACK,
    thickness=2,
)
_TRACK_LABEL = sv.LabelAnnotator(
    text_position=sv.Position.BOTTOM_CENTER,
    text_scale=0.45,
    text_thickness=1,
    border_radius=4,
    color=_TRACK_ID_PALETTE,
    color_lookup=sv.ColorLookup.TRACK,
)


def team_class_ids(teams: np.ndarray) -> np.ndarray:
    return np.where(np.isin(teams, (0, 1)), teams, 2).astype(int)


def draw_hud_bar(frame: np.ndarray, title: str, *, height: int = 44) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, height), (18, 18, 22), -1)
    frame[:] = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)
    draw_text_shadow(
        frame, title, (14, 30), font_scale=0.75, color_bgr=ROBOFLOW_PURPLE_BGR, thickness=2
    )
    return frame


def draw_branding_tag(frame: np.ndarray, text: str = "powered by trackers") -> np.ndarray:
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    pad = 8
    x0 = w - tw - pad * 2 - 10
    y0 = h - th - pad * 2 - 10
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x0 - pad, y0 - pad),
        (w - 10, h - 10),
        (18, 18, 22),
        -1,
    )
    frame[:] = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    draw_text_shadow(
        frame,
        text,
        (x0, y0 + th),
        font_scale=0.48,
        color_bgr=ROBOFLOW_PURPLE_BGR,
        thickness=1,
    )
    return frame




def draw_kalman_joystick_dots(
    frame: np.ndarray,
    dets: sv.Detections,
    *,
    min_speed_px: float = 0.5,
    max_speed_px: float = 4.0,
    dot_smoother: JoystickDotSmoother | None = None,
    speed_smoother: KalmanSpeedDisplaySmoother | None = None,
    transformer=None,
    fps: float = 25.0,
    show_speed: bool = False,
    min_speed_ms: float = 0.0,
    max_speed_labels: int = 0,
    speed_source: str = "kalman",
) -> np.ndarray:
    """PlayStation-style direction dot on each player ellipse from Kalman velocity."""
    if len(dets) == 0 or dets.data is None:
        return frame
    kf_vx = dets.data.get("kf_vx")
    kf_vy = dets.data.get("kf_vy")
    if kf_vx is None or kf_vy is None:
        return frame

    pmask = np.isin(dets.class_id, (ROLE_PLAYER, ROLE_GOALKEEPER))
    if not pmask.any():
        return frame

    from analytics.pass_imports import feet_xy
    from analytics.player_motion import kalman_ground_speed_m_s

    feet = feet_xy(dets) if show_speed else None
    teams = dets.data.get("team", np.full(len(dets), -1))
    tids = dets.tracker_id if dets.tracker_id is not None else np.full(len(dets), -1, dtype=int)
    speed_badges: list[
        tuple[float, int, int, int, int, int, float, float, tuple[int, int, int]]
    ] = []
    for i in np.flatnonzero(pmask):
        team = int(teams[i])
        if team not in (0, 1):
            continue
        cx, cy, a, b = player_ellipse_geometry(dets.xyxy[i])
        radius = _dot_radius_for_ellipse(a)
        vx, vy = float(kf_vx[i]), float(kf_vy[i])
        speed_px = 0.0
        if not np.isfinite(vx) or not np.isfinite(vy):
            px, py = cx, cy
        else:
            speed_px = float(np.hypot(vx, vy))
            if speed_px < min_speed_px:
                px, py = cx, cy
            else:
                stick = kalman_speed_stick(
                    speed_px, min_speed_px=min_speed_px, max_speed_px=max_speed_px
                )
                if stick is None:
                    px, py = cx, cy
                else:
                    ux, uy = vx / speed_px, vy / speed_px
                    reach = _joystick_dot_reach(
                        stick, a, b, ux, uy, dot_radius=float(radius)
                    )
                    px, py = cx + ux * reach, cy + uy * reach
        if dot_smoother is not None:
            px, py = dot_smoother.smooth(
                int(tids[i]), float(cx), float(cy), float(px), float(py)
            )
        else:
            px, py = int(px), int(py)
        dot_color = TEAM_COLORS[team].as_bgr()
        cv2.circle(frame, (px, py), radius, dot_color, -1, cv2.LINE_AA)
        if show_speed:
            if speed_source == "multilag":
                speed_arr = dets.data.get("speed_ms") if dets.data else None
                speed_m_s = (
                    float(speed_arr[i])
                    if speed_arr is not None and np.isfinite(speed_arr[i])
                    else 0.0
                )
                if speed_smoother is not None:
                    speed_m_s = speed_smoother.smooth(int(tids[i]), speed_m_s)
            elif feet is not None:
                if np.isfinite(vx) and np.isfinite(vy):
                    speed_m_s = kalman_ground_speed_m_s(
                        feet[i],
                        np.array([vx, vy], dtype=np.float64),
                        transformer,
                        fps=fps,
                    )
                else:
                    speed_m_s = 0.0
                if speed_m_s is not None and speed_smoother is not None:
                    speed_m_s = speed_smoother.smooth(int(tids[i]), speed_m_s)
                elif speed_m_s is None:
                    speed_m_s = 0.0
            else:
                speed_m_s = 0.0
            if speed_m_s >= min_speed_ms:
                speed_badges.append(
                    (speed_m_s, cx, cy, px, py, radius, vx, vy, dot_color)
                )

    if show_speed and speed_badges:
        if max_speed_labels > 0:
            speed_badges.sort(key=lambda item: item[0], reverse=True)
            speed_badges = speed_badges[:max_speed_labels]
        for speed_m_s, cx, cy, px, py, radius, vx, vy, color in speed_badges:
            draw_speed_badge(
                frame,
                speed_m_s,
                float(cx),
                float(cy),
                px,
                py,
                vx,
                vy,
                team_bgr=color,
                dot_radius=radius,
                min_speed_px=min_speed_px,
            )
    return frame


def _tracker_id_labels(dets: sv.Detections) -> list[str]:
    n = len(dets)
    tids = dets.tracker_id if dets.tracker_id is not None else np.full(n, -1, dtype=int)
    return [f"#{int(tid)}" if int(tid) >= 0 else "" for tid in tids]


def annotate_players(
    frame: np.ndarray,
    dets: sv.Detections,
    *,
    labels: list[str] | None = None,
    facing: np.ndarray | None = None,
    facing_motion: np.ndarray | None = None,
    facing_kalman: np.ndarray | None = None,
    show_kalman_joystick: bool = False,
    dot_smoother: JoystickDotSmoother | None = None,
    show_tracker_ids: bool = False,
) -> np.ndarray:
    outfield = dets[dets.class_id == ROLE_PLAYER]
    if len(outfield):
        teams = outfield.data.get("team", np.zeros(len(outfield)))
        outfield_vis = sv.Detections(
            xyxy=outfield.xyxy,
            class_id=team_class_ids(teams),
            tracker_id=outfield.tracker_id,
            data=outfield.data,
        )
        frame = _ELLIPSE.annotate(frame, outfield_vis)
        player_labels = labels
        if player_labels is None and show_tracker_ids:
            player_labels = _tracker_id_labels(outfield)
        if player_labels is not None:
            frame = _LABEL.annotate(frame, outfield_vis, labels=player_labels)

    gks = dets[dets.class_id == ROLE_GOALKEEPER]
    if len(gks):
        gk_teams = gks.data.get("team", np.full(len(gks), -1))
        has_team = np.isin(gk_teams, (0, 1))
        if has_team.any():
            gk_colored = gks[has_team]
            gk_vis = sv.Detections(
                xyxy=gk_colored.xyxy,
                class_id=team_class_ids(gk_teams[has_team]),
                tracker_id=gk_colored.tracker_id,
                data=gk_colored.data,
            )
            frame = _ELLIPSE.annotate(frame, gk_vis)
            if show_tracker_ids:
                frame = _LABEL.annotate(
                    frame, gk_vis, labels=_tracker_id_labels(gk_colored)
                )
        if (~has_team).any():
            gk_neutral = gks[~has_team]
            frame = _GK_ELLIPSE.annotate(frame, gk_neutral)
            if show_tracker_ids:
                gk_vis = sv.Detections(
                    xyxy=gk_neutral.xyxy,
                    class_id=np.zeros(len(gk_neutral), dtype=int),
                    tracker_id=gk_neutral.tracker_id,
                    data=gk_neutral.data,
                )
                frame = _LABEL.annotate(
                    frame, gk_vis, labels=_tracker_id_labels(gk_neutral)
                )

    if show_kalman_joystick:
        frame = draw_kalman_joystick_dots(frame, dets, dot_smoother=dot_smoother)
    return frame


def annotate_ball(frame: np.ndarray, dets: sv.Detections) -> np.ndarray:
    ball = ball_xy(dets)
    if ball is None:
        return frame
    x, y = float(ball[0]), float(ball[1])
    ball_dets = sv.Detections(
        xyxy=np.array([[x - 6, y - 6, x + 6, y + 6]], dtype=np.float32),
        class_id=np.array([0]),
    )
    return _BALL_TRI.annotate(frame, ball_dets)


def _valid_pitch_cm(
    xy: np.ndarray, config=PITCH_CONFIG, *, margin_cm: float = 200.0
) -> np.ndarray:
    """Mask for points that lie on the pitch (homography outliers are dropped)."""
    from analytics.homography import valid_pitch_cm

    return valid_pitch_cm(xy, config, margin_cm=margin_cm)


def draw_radar_minimap(
    frame: np.ndarray,
    dets: sv.Detections,
    keypoints: sv.KeyPoints | None = None,
    *,
    scale_frac: float = 0.33,
    position: str = "bottom_right",
    pitch_confidence: float = 0.9,
    use_ransac: bool = False,
    ransac_thresh: float = HOMOGRAPHY_RANSAC_REPROJ_THRESH,
    transformer: ViewTransformer | None = None,
    clip_radar_transformer: ViewTransformer | None = None,
    locked_goal_defenders: tuple[int, int] | None = None,
    prebuilt_radar: np.ndarray | None = None,
    debug_keypoints: bool = False,
    opacity: float = 0.5,
) -> np.ndarray:
    """Sports-style radar minimap: per-frame H from gated keypoints (no mirror lock)."""
    del clip_radar_transformer  # deprecated; minimap always fits per-frame from keypoints
    if prebuilt_radar is not None:
        radar = prebuilt_radar
    elif keypoints is not None:
        radar = render_radar(
            dets,
            keypoints,
            confidence=pitch_confidence,
            locked_goal_defenders=locked_goal_defenders,
            debug_keypoints=debug_keypoints,
        )
    elif transformer is not None:
        radar = render_radar(
            dets,
            None,
            transformer=transformer,
            locked_goal_defenders=locked_goal_defenders,
        )
    else:
        return frame
    if radar is None:
        return frame

    h, w, _ = frame.shape
    rw = max(int(w * scale_frac), 120)
    rh = int(radar.shape[0] * (rw / radar.shape[1]))
    radar = sv.resize_image(radar, (rw, rh))
    margin = 14
    brand_clearance = 52
    if position == "bottom_left":
        x, y = margin, h - rh - margin
    elif position == "bottom_center":
        x, y = (w - rw) // 2, h - rh - margin
    else:
        x, y = w - rw - margin, h - rh - margin - brand_clearance
    rect = sv.Rect(x=x, y=y, width=rw, height=rh)
    return sv.draw_image(frame, radar, opacity=opacity, rect=rect)


_KP_USED_BGR = (80, 220, 80)
_KP_LOW_CONF_BGR = (80, 80, 255)
_KP_INVALID_BGR = (140, 140, 140)
_KP_EDGE_BGR = (70, 70, 90)
_KP_RAW_BGR = (255, 220, 80)


def _keypoint_xy_valid(x: float, y: float) -> bool:
    return bool(np.isfinite(x) and np.isfinite(y) and x > 1 and y > 1)


def draw_pitch_keypoints_debug(
    frame: np.ndarray,
    keypoints: sv.KeyPoints | None,
    *,
    confidence_threshold: float = 0.5,
    draw_skeleton: bool = True,
    show_rejected: bool = True,
) -> np.ndarray:
    """Overlay raw pitch-keypoint detections (index + confidence) for homography debugging.

    Labels use 0-based indices matching ``config.vertices`` / the YOLO pose head order.
    Skeleton edges only connect keypoints that pass the confidence filter (model is
    trained on broadcast football; SoccerNet angles may still look sparse or wrong).
    Missing/placeholder model outputs (``x <= 1`` or ``y <= 1``) are omitted from the
    overlay so they do not stack on the frame border.
    """
    margin = 14
    legend_y = 52  # below HUD bar; avoids overlapping bottom-right radar

    if keypoints is None or keypoints.xy.shape[0] == 0:
        draw_text_shadow(
            frame,
            "pitch keypoints: none",
            (margin, legend_y),
            font_scale=0.5,
            color_bgr=(180, 180, 180),
            thickness=1,
        )
        return frame

    xy = keypoints.xy[0]
    n = len(PITCH_CONFIG.vertices)
    from analytics.homography import (
        pitch_keypoint_inlier_mask,
        view_transformer_from_keypoints,
    )

    conf = pitch_keypoint_confidence(keypoints, n_vertices=n)
    h_t = view_transformer_from_keypoints(
        keypoints, confidence=confidence_threshold, use_ransac=True
    )
    accept = pitch_keypoint_inlier_mask(
        xy, conf, h_t, confidence=confidence_threshold, max_reproj_px=8.0
    )

    n_invalid = 0
    if draw_skeleton:
        # edges are 1-based vertex ids (same as draw_pitch / roboflow sports)
        for start, end in PITCH_CONFIG.edges:
            i, j = start - 1, end - 1
            if i >= len(xy) or j >= len(xy):
                continue
            if not (
                _keypoint_xy_valid(float(xy[i, 0]), float(xy[i, 1]))
                and _keypoint_xy_valid(float(xy[j, 0]), float(xy[j, 1]))
            ):
                continue
            if not (accept[i] and accept[j]):
                continue
            p1 = (int(xy[i, 0]), int(xy[i, 1]))
            p2 = (int(xy[j, 0]), int(xy[j, 1]))
            cv2.line(frame, p1, p2, _KP_EDGE_BGR, 1, cv2.LINE_AA)

    for i in range(min(len(xy), n)):
        x, y = float(xy[i, 0]), float(xy[i, 1])
        if not _keypoint_xy_valid(x, y):
            n_invalid += 1
            continue
        if accept[i]:
            color = _KP_USED_BGR
        elif show_rejected:
            color = _KP_LOW_CONF_BGR
        else:
            continue
        px, py = int(x), int(y)
        cv2.circle(frame, (px, py), 5, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)
        label = f"{i}:{conf[i]:.2f}"
        draw_text_shadow(
            frame,
            label,
            (px + 7, py - 6),
            font_scale=0.38,
            color_bgr=color,
            thickness=1,
        )

    n_ok = int(accept[: min(len(xy), n)].sum())
    summary = (
        f"pitch kp {n_ok}/{n} used (conf > {confidence_threshold:.2f}, "
        f"{n_invalid} missing)"
    )
    draw_text_shadow(
        frame, summary, (margin, legend_y), font_scale=0.52, color_bgr=(230, 230, 230), thickness=1
    )
    for dy, text, color in (
        (22, "green = used for H (tracker upgrade path)", _KP_USED_BGR),
        (40, "red = low confidence", _KP_LOW_CONF_BGR),
        (58, "gray = invalid / missing", _KP_INVALID_BGR),
        (76, "see compare overlay for smoothed H inputs", _KP_EDGE_BGR),
    ):
        draw_text_shadow(
            frame, text, (margin, legend_y + dy), font_scale=0.42, color_bgr=color, thickness=1
        )
    return frame


def ease_out_cubic(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return 1.0 - (1.0 - t) ** 3


def draw_carrier_spotlight(
    dimmed: np.ndarray,
    original: np.ndarray,
    center: tuple[int, int],
    *,
    radius: int = 150,
    strength: float = 0.62,
) -> np.ndarray:
    """Keep the ball carrier readable while the rest of the frame is dimmed."""
    h, w = dimmed.shape[:2]
    cx, cy = center
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (cx, cy), radius, 1.0, -1, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=radius * 0.38)
    mask = (mask[..., None] * strength).astype(np.float32)
    out = dimmed.astype(np.float32) * (1.0 - mask) + original.astype(np.float32) * mask
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_carrier_halo(
    frame: np.ndarray,
    center: tuple[int, int],
    *,
    radius: int = 20,
    strength: float = 0.24,
) -> None:
    """Soft white glow at the ball carrier's feet (drawn in-place)."""
    h, w = frame.shape[:2]
    cx, cy = int(center[0]), int(center[1])
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (cx, cy), radius, 1.0, -1, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(radius * 0.42, 1.0))
    mask = (mask[..., None] * strength).astype(np.float32)
    glow = frame.astype(np.float32).copy()
    cv2.circle(glow, (cx, cy), max(radius - 4, 6), (255, 255, 255), -1, cv2.LINE_AA)
    frame[:] = np.clip(
        frame.astype(np.float32) * (1.0 - mask) + glow * mask,
        0,
        255,
    ).astype(np.uint8)


def _draw_projected_ground_zone(
    frame: np.ndarray,
    poly: np.ndarray,
    color_bgr: tuple[int, int, int],
    *,
    alpha: float,
    thickness: int,
    filled: bool,
) -> None:
    """Fill and/or outline a perspective ellipse polygon on the broadcast view."""
    if poly is None or len(poly) < 3:
        return
    if filled and alpha > 0.01:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [poly], color_bgr, lineType=cv2.LINE_AA)
        frame[:] = cv2.addWeighted(overlay, alpha * 0.45, frame, 1.0 - alpha * 0.45, 0)
    if thickness > 0 and alpha > 0.01:
        outline = frame.copy()
        cv2.polylines(
            outline,
            [poly],
            isClosed=True,
            color=color_bgr,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
        frame[:] = cv2.addWeighted(outline, alpha, frame, 1.0 - alpha, 0)


def _image_space_ground_ellipse(
    frame: np.ndarray,
    center: tuple[int, int],
    *,
    radius_px: int,
    color_bgr: tuple[int, int, int],
    alpha: float,
    thickness: int,
    filled: bool,
) -> None:
    """Fallback ellipse at the feet when homography is unavailable."""
    cx, cy = int(center[0]), int(center[1])
    axes = (max(radius_px, 8), max(radius_px // 3, 4))
    if filled and alpha > 0.01:
        overlay = frame.copy()
        cv2.ellipse(overlay, (cx, cy), axes, 0, 0, 360, color_bgr, -1, cv2.LINE_AA)
        frame[:] = cv2.addWeighted(overlay, alpha * 0.45, frame, 1.0 - alpha * 0.45, 0)
    if thickness > 0 and alpha > 0.01:
        outline = frame.copy()
        cv2.ellipse(outline, (cx, cy), axes, 0, 0, 360, color_bgr, thickness, cv2.LINE_AA)
        frame[:] = cv2.addWeighted(outline, alpha, frame, 1.0 - alpha, 0)


def draw_carrier_ground_ellipse(
    frame: np.ndarray,
    center: tuple[float, float] | np.ndarray,
    *,
    transformer: ViewTransformer | None = None,
    color_bgr: tuple[int, int, int] = ROBOFLOW_PURPLE_BGR,
    radius_m: float = 0.55,
    radius_px: int = 28,
    alpha: float = 0.7,
    thickness: int = 2,
    filled: bool = True,
    pulse_t: float | None = None,
) -> bool:
    """Highlight the carrier zone as a pitch-space circle (ellipse on the turf).

    With homography, samples a ground circle in meters and projects it through H.
    Without homography, falls back to a feet-centered image ellipse.
    """
    feet = np.asarray(center, dtype=np.float64).reshape(1, 2)
    center_px = (int(round(feet[0, 0])), int(round(feet[0, 1])))

    if pulse_t is not None:
        for i, base_r in enumerate((0.40, 0.56, 0.72)):
            phase = (pulse_t + i * 0.22) % 1.0
            wave = 0.5 + 0.5 * np.sin(phase * 2.0 * np.pi)
            ring_r = base_r * radius_m * (0.92 + 0.12 * wave)
            ring_alpha = 0.18 + 0.16 * wave
            if transformer is not None:
                center_m = image_to_pitch_m(feet, transformer)
                if center_m is not None:
                    poly = pitch_circle_to_image(center_m[0], ring_r, transformer)
                    _draw_projected_ground_zone(
                        frame,
                        poly,
                        color_bgr,
                        alpha=ring_alpha,
                        thickness=2,
                        filled=False,
                    )
                    continue
            ring_px = int(radius_px * (0.75 + 0.35 * (i + 1) / 3.0) * (0.92 + 0.12 * wave))
            _image_space_ground_ellipse(
                frame,
                center_px,
                radius_px=ring_px,
                color_bgr=color_bgr,
                alpha=ring_alpha,
                thickness=2,
                filled=False,
            )

    if transformer is not None:
        center_m = image_to_pitch_m(feet, transformer)
        if center_m is not None:
            poly = pitch_circle_to_image(center_m[0], radius_m, transformer)
            _draw_projected_ground_zone(
                frame,
                poly,
                color_bgr,
                alpha=alpha,
                thickness=thickness,
                filled=filled,
            )
            return True

    _image_space_ground_ellipse(
        frame,
        center_px,
        radius_px=radius_px,
        color_bgr=color_bgr,
        alpha=alpha,
        thickness=thickness,
        filled=filled,
    )
    return False


def draw_carrier_pulse(
    frame: np.ndarray,
    center: tuple[int, int],
    t: float,
    *,
    color_bgr: tuple[int, int, int] = ROBOFLOW_PURPLE_BGR,
) -> None:
    """Animated focus rings on the ball carrier."""
    cx, cy = center
    for i, base_r in enumerate((22, 36, 52)):
        phase = (t + i * 0.22) % 1.0
        wave = 0.5 + 0.5 * np.sin(phase * 2.0 * np.pi)
        radius = int(base_r * (0.92 + 0.12 * wave))
        alpha = 0.22 + 0.18 * wave
        layer = frame.copy()
        cv2.circle(layer, (cx, cy), radius, color_bgr, 2, cv2.LINE_AA)
        frame[:] = cv2.addWeighted(layer, alpha, frame, 1.0 - alpha, 0)
    cv2.circle(frame, (cx, cy), 10, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 14, color_bgr, 2, cv2.LINE_AA)


def draw_pass_analysis_panel(
    frame: np.ndarray,
    *,
    progress: float,
    revealed: int,
    total: int = 3,
    rank_label: str | None = None,
    rank_color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Bottom broadcast-style panel for the freeze / reveal sequence."""
    h, w = frame.shape[:2]
    panel_h = 78
    panel_w = min(460, w - 48)
    px = (w - panel_w) // 2
    py = h - panel_h - 62

    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (16, 16, 20), -1)
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (48, 48, 58), 1)
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + 4), ROBOFLOW_PURPLE_BGR, -1)
    frame[:] = cv2.addWeighted(overlay, 0.9, frame, 0.1, 0)

    if revealed == 0:
        title = "PASS ANALYSIS"
        step = int(progress * 9) % 3
        dots = "." * (step + 1)
        subtitle = f"Scanning open lanes{dots}"
    else:
        label = rank_label or f"OPTION {revealed}"
        title = label
        subtitle = f"Route {revealed} of {total}  |  ranked by lane + distance"

    draw_text_shadow(
        frame,
        title,
        (px + 18, py + 30),
        font_scale=0.62,
        color_bgr=rank_color or (255, 255, 255),
        thickness=2,
    )
    draw_text_shadow(
        frame,
        subtitle,
        (px + 18, py + 58),
        font_scale=0.46,
        color_bgr=(175, 175, 185),
        thickness=1,
    )

    bar_w = panel_w - 36
    bar_x = px + 18
    bar_y = py + panel_h - 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 4), (40, 40, 48), -1)
    fill = int(bar_w * ease_out_cubic(progress if revealed else (0.35 + 0.65 * progress)))
    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + max(fill, 6), bar_y + 4),
        rank_color or ROBOFLOW_PURPLE_BGR,
        -1,
    )
    return frame


def draw_glow_arrow(
    frame: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color_bgr: tuple[int, int, int],
    *,
    thickness: int = 4,
    alpha: float = 1.0,
) -> None:
    if alpha <= 0.01:
        return
    layer = frame.copy()
    slim = thickness <= 3
    shadow = thickness + (2 if slim else 3)
    tip_len = 0.035 if slim else 0.05
    cv2.arrowedLine(
        layer, start, end, (20, 20, 20), shadow, cv2.LINE_AA, tipLength=tip_len
    )
    cv2.arrowedLine(
        layer, start, end, color_bgr, thickness, cv2.LINE_AA, tipLength=tip_len
    )
    if alpha >= 0.99:
        frame[:] = layer
    else:
        frame[:] = cv2.addWeighted(layer, alpha, frame, 1.0 - alpha, 0)


def draw_score_chip(
    frame: np.ndarray,
    text: str,
    center: tuple[int, int],
    *,
    bg_bgr: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.5, 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    cx, cy = center
    x0, y0 = cx - tw // 2 - 8, cy - th // 2 - 6
    x1, y1 = cx + tw // 2 + 8, cy + th // 2 + baseline + 6
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), bg_bgr, -1)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 255, 255), 1)
    frame[:] = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
    draw_text_shadow(frame, text, (x0 + 8, y0 + th + 4), font_scale=scale, color_bgr=(255, 255, 255), thickness=thick)


# --- pass lane overlays ---

RANK_LABELS = ["BEST", "2ND", "3RD"]

_RADAR_SCALE = 0.1
_RADAR_PADDING = 50
_CORRIDOR_ALPHA_RADAR = 0.45
_CORRIDOR_ALPHA_VIDEO = 0.38
_CORRIDOR_ALPHA_PRESENTATION = 0.62
_RANK_COLORS = [
    sv.Color.from_hex("#3CDC3C"),  # best (Green)
    sv.Color.from_hex("#FFD700"),  # 2nd (Yellow instead of Cyan/blue shadow)
    sv.Color.from_hex("#FF8C28"),  # 3rd (Orange)
]
_RANK_BGR = [c.as_bgr() for c in _RANK_COLORS]
_BLOCKER_BGR = (40, 40, 255)
RANK_COLORS_BGR = [(80, 220, 60), (0, 215, 255), (40, 140, 255)]


def pass_line_label_xy(
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    along: float = 0.42,
    offset_px: int = 14,
) -> tuple[int, int]:
    """Point beside the pass segment for a distance label (freeze-frame style)."""
    sx, sy = start
    ex, ey = end
    px = sx + along * (ex - sx)
    py = sy + along * (ey - sy)
    dx, dy = ex - sx, ey - sy
    norm = float(np.hypot(dx, dy)) or 1.0
    return (
        int(px - dy / norm * offset_px),
        int(py + dx / norm * offset_px),
    )


def apply_pass_lane_geometry(
    frame: np.ndarray,
    options: list[PassOption],
    transformer: ViewTransformer,
    feet_xy: np.ndarray,
    *,
    display_ranks: list[int] | None = None,
) -> np.ndarray:
    """Shaded corridors + rival blockers — same as pass-alternatives freeze video."""
    out = draw_pass_corridors_on_frame(
        frame, options, transformer, display_ranks=display_ranks
    )
    return draw_blocking_rivals_on_frame(out, options, feet_xy=feet_xy)


def draw_receiver_highlight(
    frame: np.ndarray,
    center: tuple[int, int],
    rank: int,
    color_bgr: tuple[int, int, int],
    *,
    alpha: float = 1.0,
) -> None:
    """Ring at the receiver feet (BEST gets a white outer ring)."""
    rx, ry = center
    ring = 20 if rank == 0 else 14
    layer = frame.copy()
    cv2.circle(layer, (rx, ry), ring, color_bgr, 3 if rank == 0 else 2, cv2.LINE_AA)
    if rank == 0:
        cv2.circle(layer, (rx, ry), ring + 6, (255, 255, 255), 1, cv2.LINE_AA)
    if alpha >= 0.99:
        frame[:] = layer
    else:
        frame[:] = cv2.addWeighted(layer, alpha, frame, 1.0 - alpha, 0)


def pitch_cm_to_radar_px(
    points_cm: np.ndarray,
    *,
    config: SoccerPitchConfiguration = PITCH_CONFIG,
    scale: float = _RADAR_SCALE,
    padding: int = _RADAR_PADDING,
) -> np.ndarray:
    pts = np.asarray(points_cm, dtype=np.float64).reshape(-1, 2)
    return np.stack(
        [
            (pts[:, 0] * scale + padding).astype(np.int32),
            (pts[:, 1] * scale + padding).astype(np.int32),
        ],
        axis=1,
    )


def _draw_blocker_markers(
    out: np.ndarray,
    pitch_cm: np.ndarray,
    indices: tuple[int, ...],
    *,
    config: SoccerPitchConfiguration,
    drawn: set[int] | None = None,
) -> set[int]:
    """Small red badge with ! — no white halo (matches freeze video)."""
    seen = drawn if drawn is not None else set()
    for idx in indices:
        if idx in seen or idx >= len(pitch_cm):
            continue
        seen.add(idx)
        pt = pitch_cm_to_radar_px(pitch_cm[idx : idx + 1], config=config)[0]
        cv2.circle(out, tuple(pt), 12, _BLOCKER_BGR, -1, cv2.LINE_AA)
        cv2.putText(
            out,
            "!",
            (pt[0] - 5, pt[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return seen


def draw_pass_lanes_on_radar(
    radar: np.ndarray,
    options: list[PassOption],
    pitch_cm: np.ndarray,
    *,
    config: SoccerPitchConfiguration = PITCH_CONFIG,
    display_ranks: list[int] | None = None,
) -> np.ndarray:
    """Overlay ranked pass corridors and highlight blocking rivals (pitch cm coords)."""
    out = radar.copy()
    drawn: set[int] = set()
    for idx, option in enumerate(options):
        rank = display_ranks[idx] if display_ranks is not None else idx
        debug = option.lane_debug
        if debug is None:
            continue
        color = _RANK_COLORS[min(rank, len(_RANK_COLORS) - 1)]
        poly = pitch_cm_to_radar_px(debug.corridor_polygon_cm, config=config)
        overlay = out.copy()
        cv2.fillPoly(overlay, [poly], color.as_bgr())
        out = cv2.addWeighted(overlay, _CORRIDOR_ALPHA_RADAR, out, 1.0 - _CORRIDOR_ALPHA_RADAR, 0)
        cv2.polylines(out, [poly], isClosed=True, color=color.as_bgr(), thickness=2)

        drawn = _draw_blocker_markers(
            out,
            pitch_cm,
            debug.blocking_rival_indices,
            config=config,
            drawn=drawn,
        )

    return out


def _corridor_image_polygon(
    corridor_polygon_cm: np.ndarray,
    transformer: ViewTransformer,
    frame_shape: tuple[int, int],
) -> np.ndarray | None:
    """Project a pitch corridor (cm) onto image pixels, clipped to the frame."""
    img = pitch_cm_to_image(corridor_polygon_cm, transformer)
    if img is None or len(img) < 3:
        return None
    h, w = frame_shape[:2]
    poly = img.astype(np.int32)
    if not ((poly[:, 0] >= -w) & (poly[:, 0] <= 2 * w) & (poly[:, 1] >= -h) & (poly[:, 1] <= 2 * h)).any():
        return None
    return poly


def draw_pass_corridors_on_frame(
    frame: np.ndarray,
    options: list[PassOption],
    transformer: ViewTransformer,
    *,
    presentation: bool = False,
    display_ranks: list[int] | None = None,
) -> np.ndarray:
    """Draw pitch-space pass corridors on the main camera view (H^-1)."""
    out = frame
    alpha = _CORRIDOR_ALPHA_PRESENTATION if presentation else _CORRIDOR_ALPHA_VIDEO
    thickness = 5 if presentation else 3
    for idx, option in enumerate(options):
        rank = display_ranks[idx] if display_ranks is not None else idx
        debug = option.lane_debug
        if debug is None:
            continue
        poly = _corridor_image_polygon(
            debug.corridor_polygon_cm, transformer, out.shape
        )
        if poly is None:
            continue
        color = _RANK_BGR[min(rank, len(_RANK_BGR) - 1)]
        overlay = out.copy()
        cv2.fillPoly(overlay, [poly], color)
        out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0)
        cv2.polylines(out, [poly], isClosed=True, color=color, thickness=thickness)

    return out


def draw_blocking_rivals_on_frame(
    frame: np.ndarray,
    options: list[PassOption],
    *,
    feet_xy: np.ndarray,
) -> np.ndarray:
    """Mark rivals counted inside a corridor — compact red ! at feet."""
    out = frame
    drawn: set[int] = set()
    for option in options:
        debug = option.lane_debug
        if debug is None:
            continue
        for idx in debug.blocking_rival_indices:
            if idx in drawn or idx >= len(feet_xy):
                continue
            drawn.add(idx)
            x, y = int(feet_xy[idx, 0]), int(feet_xy[idx, 1])
            cv2.circle(out, (x, y), 14, _BLOCKER_BGR, -1, cv2.LINE_AA)
            cv2.putText(
                out,
                "!",
                (x - 7, y + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    return out


def draw_pass_lane_legend(frame: np.ndarray) -> np.ndarray:
    """Legend for pass-lane debug overlays."""
    lines = [
        "shaded = pass corridor (pitch/radar; wider on long passes)",
        "green/cyan/orange = BEST / 2ND / 3RD",
        "red ! = rival inside corridor",
    ]
    y = 78
    for line in lines:
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        y += 18
    return frame


def _options_with_lane_debug(
    dets: sv.Detections,
    event: "PassEvent",
    *,
    weights,
    transformer: ViewTransformer,
    pitch_cm: np.ndarray | None = None,
) -> list[PassOption]:
    """Re-score if needed so freeze frames always carry pitch corridor geometry."""
    from analytics.pass_imports import PassWeights, bbox_center_xy, remap_lane_debug_to_pitch_cm, top_pass_options

    if event.options and all(o.lane_debug is not None for o in event.options):
        return event.options
    pitch_feet = image_to_pitch_m(feet_xy(dets), transformer)
    if pitch_cm is None:
        pitch_cm = image_to_pitch_cm(feet_xy(dets), transformer)
    body_pitch_m = image_to_pitch_m(bbox_center_xy(dets), transformer)
    if pitch_feet is None or pitch_cm is None:
        return event.options
    attack_dir = pitch_attack_direction(
        dets,
        event.carrier.team,
        transformer,
        player_mask_fn=player_mask,
        feet_fn=feet_xy,
    )
    return top_pass_options(
        dets,
        event.carrier,
        k=3,
        weights=weights,
        attack_dir=attack_dir,
        positions=pitch_feet,
        pitch_cm=pitch_cm,
        body_pitch_m=body_pitch_m,
    )


def draw_pass_overlay(
    frame: np.ndarray,
    dets: sv.Detections,
    event: "PassEvent",
    *,
    weights=None,
    metric: bool = False,
    keypoints: sv.KeyPoints | None = None,
    pitch_confidence: float = 0.9,
    transformer: ViewTransformer | None = None,
    show_lane_debug: bool = True,
    show_radar: bool = True,
    locked_goal_defenders: tuple[int, int] | None = None,
    debug_pitch_keypoints: bool = False,
    revealed_options: int | None = None,
    reveal_progress: float = 1.0,
    facing: np.ndarray | None = None,
    facing_motion: np.ndarray | None = None,
    facing_kalman: np.ndarray | None = None,
    show_kalman_joystick: bool = False,
    dot_smoother=None,
) -> np.ndarray:
    """Dim the frame and draw ranked pass arrows from the carrier.

    ``revealed_options``: how many top options to show (0 = carrier only, None = all).
    ``reveal_progress``: 0-1 animation within the current reveal phase.
    """
    from analytics.homography import homography_from_keypoints_radar
    from analytics.pass_imports import PassWeights, remap_lane_debug_to_pitch_cm

    if weights is None:
        weights = PassWeights()

    dim = (frame.astype(np.float32) * 0.32).astype(np.uint8)
    if debug_pitch_keypoints and keypoints is not None:
        dim = draw_pitch_keypoints_debug(
            dim, keypoints, confidence_threshold=pitch_confidence
        )

    options = event.options
    feet_img = feet_xy(dets)
    if show_lane_debug and transformer is not None:
        pitch_cm_vis = image_to_pitch_cm(feet_img, transformer)
        options = _options_with_lane_debug(
            dets,
            event,
            weights=weights,
            transformer=transformer,
            pitch_cm=pitch_cm_vis,
        )

    visible = options
    if revealed_options is not None:
        visible = options[: max(0, revealed_options)]

    if show_lane_debug and transformer is not None and visible:
        dim = apply_pass_lane_geometry(dim, visible, transformer, feet_img)

    dim = annotate_players(
        dim,
        dets,
        facing=facing,
        facing_motion=facing_motion,
        facing_kalman=facing_kalman,
        show_kalman_joystick=show_kalman_joystick,
        dot_smoother=dot_smoother,
        show_tracker_ids=True,
    )
    dim = annotate_ball(dim, dets)

    feet = feet_xy(dets)
    carrier_xy = feet[event.carrier.index]
    cx, cy = int(carrier_xy[0]), int(carrier_xy[1])
    dim = draw_carrier_spotlight(dim, frame, (cx, cy))

    n_total = min(3, len(options))
    phase_revealed = 0 if revealed_options == 0 else revealed_options

    if revealed_options == 0:
        draw_carrier_pulse(dim, (cx, cy), reveal_progress)
        draw_score_chip(dim, "ON BALL", (cx, cy - 42), bg_bgr=ROBOFLOW_PURPLE_BGR)
        dim = draw_pass_analysis_panel(
            dim,
            progress=reveal_progress,
            revealed=0,
            total=n_total,
        )
        dim = draw_hud_bar(dim, "PASS ALTERNATIVES")
        return draw_branding_tag(dim)

    draw_carrier_pulse(dim, (cx, cy), min(1.0, reveal_progress * 0.35 + 0.65))

    for rank, option in enumerate(visible):
        color = RANK_COLORS_BGR[rank]
        recv_xy = feet[option.receiver_index]
        rx, ry = int(recv_xy[0]), int(recv_xy[1])
        is_new = rank == len(visible) - 1
        alpha = ease_out_cubic(reveal_progress) if is_new else 1.0
        draw_glow_arrow(dim, (cx, cy), (rx, ry), color, thickness=5, alpha=alpha)
        if alpha > 0.2:
            draw_receiver_highlight(dim, (rx, ry), rank, color, alpha=alpha)
        if alpha < 0.85:
            continue
        midx, midy = (cx + rx) // 2, (cy + ry) // 2
        chip = f"{RANK_LABELS[rank]}  {option.score:.2f}"
        if metric:
            chip += f"  {option.length:.1f} m"
        if option.rivals_in_lane:
            chip += f"  ({option.rivals_in_lane} riv)"
        if option.teammates_in_lane:
            chip += f"  ({option.teammates_in_lane} tm)"
        if option.lane_debug and option.lane_debug.blocking_rival_indices:
            chip += f"  !{len(option.lane_debug.blocking_rival_indices)}"
        draw_score_chip(dim, chip, (midx, midy), bg_bgr=color)
        if metric:
            lx, ly = pass_line_label_xy((cx, cy), (rx, ry))
            draw_text_shadow(
                dim,
                f"{option.length:.1f} m",
                (lx - 18, ly - 6),
                font_scale=0.58,
                color_bgr=color,
                thickness=2,
            )

    latest_rank = len(visible) - 1
    dim = draw_pass_analysis_panel(
        dim,
        progress=reveal_progress,
        revealed=phase_revealed,
        total=n_total,
        rank_label=RANK_LABELS[latest_rank] if visible else None,
        rank_color=RANK_COLORS_BGR[latest_rank] if visible else None,
    )

    if show_radar and keypoints is not None:
        radar_h = homography_from_keypoints_radar(
            keypoints, confidence=pitch_confidence
        )
        radar = render_radar(
            dets,
            keypoints,
            confidence=pitch_confidence,
            transformer=radar_h,
            locked_goal_defenders=locked_goal_defenders,
            debug_keypoints=True,
        )
        if radar is not None and show_lane_debug and visible and radar_h is not None:
            pitch_cm_radar = image_to_pitch_cm(feet_img, radar_h)
            if pitch_cm_radar is not None:
                radar_visible = remap_lane_debug_to_pitch_cm(
                    visible,
                    event.carrier,
                    pitch_cm_radar,
                    feet_img,
                    weights=weights,
                )
                radar = draw_pass_lanes_on_radar(
                    radar, radar_visible, pitch_cm_radar
                )
            dim = draw_pass_lane_legend(dim)
        if radar is not None:
            dim = draw_radar_minimap(
                dim,
                dets,
                keypoints,
                pitch_confidence=pitch_confidence,
                locked_goal_defenders=locked_goal_defenders,
                prebuilt_radar=radar,
            )

    dim = draw_hud_bar(dim, "PASS ALTERNATIVES  -  top 3 open lanes")
    return draw_branding_tag(dim)


