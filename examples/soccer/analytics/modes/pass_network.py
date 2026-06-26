"""Pass network: collaboration aggregation, overlays, and the PASS_NETWORK runner."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

import cv2
import numpy as np
import supervision as sv

from analytics.annotations import (
    annotate_ball,
    annotate_players,
    draw_branding_tag,
    draw_carrier_ground_ellipse,
    draw_hud_bar,
    draw_radar_minimap,
)
from analytics.clip_pipeline import ClipAnalysis, compute_clip_analysis, _clip_stem
from analytics.passing import InferredPass, InferredTurnover


@dataclass(frozen=True)
class CollaborationLink:
    """Directed pass count between two teammates."""

    passer_tid: int
    receiver_tid: int
    team: int
    count: int
    avg_quality: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PlayerPassSummary:
    """Pass activity for one tracked player."""

    tracker_id: int
    team: int
    passes_made: int
    passes_received: int
    avg_quality_made: float | None
    avg_quality_received: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PassNetwork:
    """Full v1 pass-interaction snapshot for a sequence."""

    sequence: str
    metric: bool
    n_passes: int
    n_turnovers: int
    passes: tuple[InferredPass, ...]
    turnovers: tuple[InferredTurnover, ...]
    links: tuple[CollaborationLink, ...]
    players: tuple[PlayerPassSummary, ...]

    def to_dict(self) -> dict:
        return {
            "sequence": self.sequence,
            "metric": self.metric,
            "n_passes": self.n_passes,
            "n_turnovers": self.n_turnovers,
            "passes": [p.to_dict() for p in self.passes],
            "turnovers": [t.to_dict() for t in self.turnovers],
            "collaboration_links": [link.to_dict() for link in self.links],
            "player_summaries": [player.to_dict() for player in self.players],
            "top_collaborators": [
                link.to_dict() for link in sorted(self.links, key=lambda l: l.count, reverse=True)[:10]
            ],
        }


def build_collaboration_links(events: list[InferredPass]) -> list[CollaborationLink]:
    """Count directed A -> B passes and average lane quality per link."""
    counts: dict[tuple[int, int, int], int] = defaultdict(int)
    quality_sums: dict[tuple[int, int, int], float] = defaultdict(float)
    quality_counts: dict[tuple[int, int, int], int] = defaultdict(int)

    for event in events:
        key = (event.passer_tid, event.receiver_tid, event.team)
        counts[key] += 1
        if event.quality_score is not None:
            quality_sums[key] += event.quality_score
            quality_counts[key] += 1

    links: list[CollaborationLink] = []
    for (passer, receiver, team), count in counts.items():
        scored = quality_counts[(passer, receiver, team)]
        links.append(
            CollaborationLink(
                passer_tid=passer,
                receiver_tid=receiver,
                team=team,
                count=count,
                avg_quality=(
                    quality_sums[(passer, receiver, team)] / scored
                    if scored
                    else None
                ),
            )
        )
    links.sort(key=lambda link: link.count, reverse=True)
    return links


def strongest_collaboration_pair(
    links: list[CollaborationLink] | tuple[CollaborationLink, ...],
) -> tuple[int, int, int, int] | None:
    """Undirected pair with the most passes between them: (tid_a, tid_b, team, count)."""
    totals: dict[tuple[int, int, int], int] = defaultdict(int)
    for link in links:
        a, b = sorted((link.passer_tid, link.receiver_tid))
        totals[(a, b, link.team)] += link.count
    if not totals:
        return None
    (a, b, team), count = max(totals.items(), key=lambda item: item[1])
    return a, b, team, count


def build_player_summaries(events: list[InferredPass]) -> list[PlayerPassSummary]:
    """Per-player pass counts and average quality as passer vs receiver."""
    teams: dict[int, int] = {}
    made_counts: dict[int, int] = defaultdict(int)
    recv_counts: dict[int, int] = defaultdict(int)
    made_quality: dict[int, list[float | None]] = defaultdict(list)
    recv_quality: dict[int, list[float | None]] = defaultdict(list)

    for event in events:
        teams[event.passer_tid] = event.team
        teams[event.receiver_tid] = event.team
        made_counts[event.passer_tid] += 1
        recv_counts[event.receiver_tid] += 1
        if event.quality_score is not None:
            made_quality[event.passer_tid].append(event.quality_score)
        if event.quality_score is not None:
            recv_quality[event.receiver_tid].append(event.quality_score)

    summaries: list[PlayerPassSummary] = []
    for tid in sorted(teams):
        summaries.append(
            PlayerPassSummary(
                tracker_id=tid,
                team=teams[tid],
                passes_made=made_counts[tid],
                passes_received=recv_counts[tid],
                avg_quality_made=_mean(made_quality[tid]),
                avg_quality_received=_mean(recv_quality[tid]),
            )
        )
    summaries.sort(key=lambda row: row.passes_made + row.passes_received, reverse=True)
    return summaries


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def build_pass_network(
    sequence_name: str,
    events: list[InferredPass],
    turnovers: list[InferredTurnover] | None = None,
    *,
    metric: bool,
) -> PassNetwork:
    """Build the v1 collaboration snapshot from inferred pass events."""
    turnovers = turnovers or []
    links = build_collaboration_links(events)
    players = build_player_summaries(events)
    return PassNetwork(
        sequence=sequence_name,
        metric=metric,
        n_passes=len(events),
        n_turnovers=len(turnovers),
        passes=tuple(events),
        turnovers=tuple(turnovers),
        links=tuple(links),
        players=tuple(players),
    )


from analytics.annotations import (
    ROBOFLOW_PURPLE_BGR,
    TEAM_COLORS,
    draw_glow_arrow,
    draw_score_chip,
    draw_text_shadow,
    ease_out_cubic,
)
from analytics.passing import ball_xy, passes_for_overlay
from analytics.player_motion import feet_xy, open_video

TEAM_COLORS_BGR = [c.as_bgr() for c in TEAM_COLORS[:2]]
NEUTRAL_BGR = (200, 200, 200)
TURNOVER_ARROW_BGR = (48, 168, 255)
TURNOVER_ACCENT_BGR = TURNOVER_ARROW_BGR
CARRIER_SHADOW_BGR = (228, 228, 228)


def _team_color(team: int) -> tuple[int, int, int]:
    return TEAM_COLORS_BGR[team] if team in (0, 1) else NEUTRAL_BGR


def _get_player_box(dets: sv.Detections, tid: int) -> np.ndarray | None:
    if dets.tracker_id is None:
        return None
    idx = np.flatnonzero(dets.tracker_id == tid)
    if len(idx) == 0:
        return None
    return dets.xyxy[idx[0]]


def _get_player_feet(dets: sv.Detections, tid: int) -> np.ndarray | None:
    if dets.tracker_id is None:
        return None
    idx = np.flatnonzero(dets.tracker_id == tid)
    if len(idx) == 0:
        return None
    return feet_xy(dets)[idx[0]]


def _draw_ground_highlight(
    image: np.ndarray,
    box: np.ndarray,
    color_bgr: tuple[int, int, int],
    *,
    alpha: float = 1.0,
    scale: float = 1.0,
    transformer=None,
) -> None:
    x0, y0, x1, y1 = box
    feet = np.array([(x0 + x1) * 0.5, y1], dtype=np.float64)
    radius_m = 0.45 * scale
    radius_px = max(int((x1 - x0) * 0.35 * scale), 10)
    draw_carrier_ground_ellipse(
        image,
        feet,
        transformer=transformer,
        color_bgr=color_bgr,
        radius_m=radius_m,
        radius_px=radius_px,
        alpha=alpha,
        thickness=2,
        filled=True,
    )


def _draw_pass_highlights(
    image: np.ndarray,
    dets: sv.Detections,
    frame_idx: int,
    passes: tuple[InferredPass, ...],
    frame_rate: float,
    frame_keypoints: dict | None = None,
    frame_transforms: dict | None = None,
    *,
    draw_player_halos: bool = True,
    transformer=None,
) -> None:
    passes = passes_for_overlay(passes)

    in_flight = [
        p
        for p in passes
        if p.frame_idx <= frame_idx <= p.frame_idx + p.gap_frames
    ]
    if len(in_flight) > 1:
        in_flight.sort(key=lambda p: (p.frame_idx, -(p.quality_score or 0.0)))
        in_flight = in_flight[:1]
    highlight_passes = tuple(in_flight) if in_flight else passes

    for p in highlight_passes:
        receive_idx = p.frame_idx + p.gap_frames
        twinkle_duration = int(frame_rate * 0.5)
        end_idx = receive_idx + twinkle_duration

        if p.frame_idx <= frame_idx <= end_idx:
            passer_box = _get_player_box(dets, p.passer_tid)
            receiver_box = _get_player_box(dets, p.receiver_tid)
            color = _team_color(p.team)

            if frame_idx <= receive_idx:
                t = (frame_idx - p.frame_idx) / p.gap_frames if p.gap_frames > 0 else 1.0
                pulse_alpha = 0.5 + 0.3 * np.sin(t * np.pi * 4)

                if draw_player_halos:
                    if passer_box is not None:
                        _draw_ground_highlight(
                            image, passer_box, color, alpha=pulse_alpha, transformer=transformer
                        )
                    if receiver_box is not None:
                        _draw_ground_highlight(
                            image, receiver_box, color, alpha=pulse_alpha, transformer=transformer
                        )

                if passer_box is not None and receiver_box is not None:
                    p_feet = (int((passer_box[0] + passer_box[2]) / 2), int(passer_box[3]))
                    r_feet = (int((receiver_box[0] + receiver_box[2]) / 2), int(receiver_box[3]))
                    ball_pos = ball_xy(dets)
                    expected_x = p_feet[0] + (r_feet[0] - p_feet[0]) * t
                    expected_y = p_feet[1] + (r_feet[1] - p_feet[1]) * t

                    if ball_pos is not None:
                        dist_expected_to_recv = np.hypot(
                            r_feet[0] - expected_x, r_feet[1] - expected_y
                        )
                        dist_ball_to_recv = np.hypot(
                            r_feet[0] - ball_pos[0], r_feet[1] - ball_pos[1]
                        )
                        if dist_ball_to_recv <= dist_expected_to_recv + 20:
                            current_tip_x = int(ball_pos[0])
                            current_tip_y = int(ball_pos[1])
                        else:
                            current_tip_x = int(expected_x)
                            current_tip_y = int(expected_y)
                    else:
                        current_tip_x = int(expected_x)
                        current_tip_y = int(expected_y)

                    damped_origin_x = int(
                        p_feet[0] * (1 - t)
                        + (current_tip_x - (r_feet[0] - p_feet[0]) * t) * t
                    )
                    damped_origin_y = int(
                        p_feet[1] * (1 - t)
                        + (current_tip_y - (r_feet[1] - p_feet[1]) * t) * t
                    )
                    damped_origin = (damped_origin_x, damped_origin_y)
                    current_tip = (current_tip_x, current_tip_y)

                    if np.hypot(
                        current_tip_x - damped_origin_x, current_tip_y - damped_origin_y
                    ) > 5:
                        draw_glow_arrow(
                            image, damped_origin, current_tip, color, alpha=0.35
                        )

            elif frame_idx <= end_idx and draw_player_halos:
                if receiver_box is not None:
                    prog = (frame_idx - receive_idx) / twinkle_duration
                    twinkle_val = np.sin(prog * 2 * 2 * np.pi)
                    twinkle_alpha = max(0.0, twinkle_val)
                    twinkle_scale = 1.0 + 0.4 * twinkle_alpha
                    _draw_ground_highlight(
                        image,
                        receiver_box,
                        color,
                        alpha=twinkle_alpha * 0.9,
                        scale=twinkle_scale,
                        transformer=transformer,
                    )


def _draw_turnover_notice(
    image: np.ndarray,
    dets: sv.Detections,
    turnover: InferredTurnover,
    frame_idx: int,
    frame_rate: float,
    *,
    transformer=None,
) -> None:
    hold_frames = max(10, int(round(1.2 * frame_rate)))
    start = turnover.interception_frame
    if frame_idx < start or frame_idx > start + hold_frames:
        return

    elapsed = frame_idx - start
    fade = 1.0 - ease_out_cubic(min(1.0, elapsed / max(hold_frames, 1)))
    h, w = image.shape[:2]

    passer_box = _get_player_box(dets, turnover.passer_tid)
    interceptor_box = _get_player_box(dets, turnover.interceptor_tid)
    pulse = 0.5 + 0.3 * np.sin(elapsed / max(frame_rate / 5, 1) * np.pi)

    if passer_box is not None:
        _draw_ground_highlight(
            image,
            passer_box,
            _team_color(turnover.passer_team),
            alpha=0.42 * fade,
            transformer=transformer,
        )
    if interceptor_box is not None:
        _draw_ground_highlight(
            image,
            interceptor_box,
            _team_color(turnover.interceptor_team),
            alpha=pulse * fade,
            scale=1.08,
            transformer=transformer,
        )
        feet = _get_player_feet(dets, turnover.interceptor_tid)
        if feet is not None:
            draw_carrier_ground_ellipse(
                image,
                feet,
                transformer=transformer,
                color_bgr=TURNOVER_ARROW_BGR,
                radius_m=0.52,
                alpha=pulse * fade * 0.85,
                pulse_t=min(1.0, elapsed / max(frame_rate * 0.2, 1)),
            )

    if passer_box is not None and interceptor_box is not None:
        p_feet = (int((passer_box[0] + passer_box[2]) / 2), int(passer_box[3]))
        r_feet = (int((interceptor_box[0] + interceptor_box[2]) / 2), int(interceptor_box[3]))
        draw_glow_arrow(
            image,
            p_feet,
            r_feet,
            TURNOVER_ARROW_BGR,
            thickness=3,
            alpha=0.4 * fade,
        )

    chip_y = h - 36
    draw_score_chip(
        image,
        f"TURNOVER  #{turnover.passer_tid} → #{turnover.interceptor_tid}",
        (w // 2, chip_y),
        bg_bgr=(18, 18, 22),
    )


def _draw_collaboration_web(
    image: np.ndarray,
    dets: sv.Detections,
    frame_idx: int,
    passes: tuple[InferredPass, ...],
) -> None:
    """Draw a dynamic network graph on the pitch representing completed passes."""
    connections: dict[tuple[int, int], dict] = {}
    for p in passes:
        receive_idx = p.frame_idx + p.gap_frames
        if receive_idx <= frame_idx:
            pair = tuple(sorted([p.passer_tid, p.receiver_tid]))
            if pair not in connections:
                connections[pair] = {"count": 0, "team": p.team}
            connections[pair]["count"] += 1

    if not connections:
        return

    overlay = image.copy()
    max_count = max(c["count"] for c in connections.values())

    for (t1, t2), data in connections.items():
        box1 = _get_player_box(dets, t1)
        box2 = _get_player_box(dets, t2)

        if box1 is not None and box2 is not None:
            p1 = (int((box1[0] + box1[2]) / 2), int(box1[3]))
            p2 = (int((box2[0] + box2[2]) / 2), int(box2[3]))

            count = data["count"]
            intensity = min(count / max(max_count, 1), 1.0)

            alpha = 0.15 + (intensity * 0.45)
            thickness = 1 + int(intensity * 3)
            color = _team_color(data["team"])

            temp_overlay = image.copy()
            cv2.line(temp_overlay, p1, p2, color, thickness, cv2.LINE_AA)
            cv2.addWeighted(temp_overlay, alpha, overlay, 1.0 - alpha, 0, overlay)

    image[:] = overlay


def _player_teams(network: PassNetwork) -> dict[int, int]:
    teams: dict[int, int] = {}
    for player in network.players:
        teams[player.tracker_id] = player.team
    for link in network.links:
        teams.setdefault(link.passer_tid, link.team)
        teams.setdefault(link.receiver_tid, link.team)
    return teams


def _collaboration_graph_panel(
    card: np.ndarray,
    network: PassNetwork,
    *,
    origin: tuple[int, int],
    size: tuple[int, int],
) -> None:
    """Draw directed pass links as a node graph (mutates ``card`` in place)."""
    ox, oy = origin
    gw, gh = size
    if gw < 80 or gh < 80:
        return

    overlay = card.copy()
    cv2.rectangle(overlay, (ox, oy), (ox + gw, oy + gh), (28, 28, 34), -1)
    cv2.rectangle(overlay, (ox, oy), (ox + gw, oy + gh), (55, 55, 65), 1)
    card[:] = cv2.addWeighted(overlay, 0.92, card, 0.08, 0)

    draw_text_shadow(
        card,
        "COLLABORATION GRAPH",
        (ox + 14, oy + 28),
        font_scale=0.55,
        color_bgr=(200, 200, 210),
        thickness=1,
    )

    if not network.links:
        draw_text_shadow(
            card,
            "no inferred links",
            (ox + 14, oy + gh // 2),
            font_scale=0.5,
            color_bgr=(140, 140, 150),
            thickness=1,
        )
        return

    teams = _player_teams(network)
    nodes = sorted(
        {link.passer_tid for link in network.links}
        | {link.receiver_tid for link in network.links}
    )
    cx, cy = ox + gw // 2, oy + gh // 2 + 12
    radius = min(gw, gh) * 0.34
    positions: dict[int, tuple[int, int]] = {}
    for i, tid in enumerate(nodes):
        angle = 2.0 * np.pi * i / max(len(nodes), 1) - np.pi / 2
        positions[tid] = (
            int(cx + radius * np.cos(angle)),
            int(cy + radius * np.sin(angle)),
        )

    max_count = max(link.count for link in network.links)
    for link in network.links:
        p0 = positions.get(link.passer_tid)
        p1 = positions.get(link.receiver_tid)
        if p0 is None or p1 is None:
            continue
        thickness = 1 + int(3 * link.count / max(max_count, 1))
        color = _team_color(link.team)
        cv2.arrowedLine(
            card, p0, p1, (20, 20, 24), thickness + 2, cv2.LINE_AA, tipLength=0.22
        )
        cv2.arrowedLine(
            card, p0, p1, color, thickness, cv2.LINE_AA, tipLength=0.22
        )
        mid = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
        draw_text_shadow(
            card,
            str(link.count),
            (mid[0] - 6, mid[1] - 4),
            font_scale=0.38,
            color_bgr=(230, 230, 230),
            thickness=1,
        )

    for tid, pos in positions.items():
        team = teams.get(tid, -1)
        color = _team_color(team)
        cv2.circle(card, pos, 16, (16, 16, 20), -1, cv2.LINE_AA)
        cv2.circle(card, pos, 16, color, 2, cv2.LINE_AA)
        draw_text_shadow(
            card,
            f"#{tid}",
            (pos[0] - 14, pos[1] + 5),
            font_scale=0.42,
            color_bgr=(255, 255, 255),
            thickness=1,
        )


def _format_quality(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def stats_end_card(size: tuple[int, int], network: PassNetwork) -> np.ndarray:
    """Full-frame summary of collaboration links and player pass volume."""
    w, h = size
    card = np.full((h, w, 3), 22, np.uint8)
    cv2.rectangle(card, (0, 0), (w, 6), ROBOFLOW_PURPLE_BGR, -1)

    draw_text_shadow(
        card,
        "PASS NETWORK",
        (40, 72),
        font_scale=1.1,
        color_bgr=ROBOFLOW_PURPLE_BGR,
        thickness=2,
    )
    mode = "metric lanes" if network.metric else "image-space lanes"
    draw_text_shadow(
        card,
        f"{network.n_passes} passes  |  {network.n_turnovers} turnovers  |  {mode}",
        (42, 108),
        font_scale=0.55,
        color_bgr=(170, 170, 170),
        thickness=1,
    )

    y_links = 170
    if network.turnovers:
        draw_text_shadow(
            card,
            "TURNOVERS",
            (40, y_links),
            font_scale=0.72,
            color_bgr=TURNOVER_ARROW_BGR,
            thickness=2,
        )
        for rank, turnover in enumerate(network.turnovers[:4], 1):
            line = (
                f"{rank}.  #{turnover.passer_tid} \u2192 #{turnover.interceptor_tid}"
                f"   (f{turnover.release_frame}\u2013{turnover.interception_frame})"
            )
            draw_text_shadow(
                card,
                line,
                (44, y_links + rank * 34),
                font_scale=0.58,
                color_bgr=(210, 210, 215),
                thickness=1,
            )
        y_links += 4 * 34 + 28

    draw_text_shadow(
        card,
        "TOP COLLABORATORS",
        (40, y_links),
        font_scale=0.72,
        color_bgr=(120, 230, 120),
        thickness=2,
    )
    for rank, link in enumerate(network.links[:6], 1):
        color = _team_color(link.team)
        line = (
            f"{rank}.  #{link.passer_tid} -> #{link.receiver_tid}"
            f"   {link.count} passes  |  Avg Quality: {_format_quality(link.avg_quality)}"
        )
        draw_text_shadow(
            card,
            line,
            (44, y_links + rank * 38),
            font_scale=0.62,
            color_bgr=color,
            thickness=2,
        )

    y_players = y_links + 8 * 38 + 20
    draw_text_shadow(
        card,
        "MOST ACTIVE PLAYERS",
        (40, y_players),
        font_scale=0.72,
        color_bgr=(40, 220, 240),
        thickness=2,
    )
    for rank, player in enumerate(network.players[:6], 1):
        color = _team_color(player.team)
        line = (
            f"{rank}.  #{player.tracker_id}"
            f"   Made: {player.passes_made}  |  Received: {player.passes_received}"
            f"  |  Avg Quality Made: {_format_quality(player.avg_quality_made)}"
        )
        draw_text_shadow(
            card,
            line,
            (44, y_players + rank * 38),
            font_scale=0.62,
            color_bgr=color,
            thickness=2,
        )

    graph_x = int(w * 0.52)
    _collaboration_graph_panel(
        card,
        network,
        origin=(graph_x, 150),
        size=(w - graph_x - 40, h - 220),
    )

    strongest = strongest_collaboration_pair(network.links)
    if strongest is not None:
        tid_a, tid_b, _team, count = strongest
        draw_text_shadow(
            card,
            f"STRONGEST LINK:  #{tid_a} <-> #{tid_b}  ({count} passes)",
            (44, h - 80),
            font_scale=0.75,
            color_bgr=(120, 230, 120),
            thickness=2,
        )

    return draw_branding_tag(card)


def draw_pass_network_frame_overlays(
    image: np.ndarray,
    dets: sv.Detections,
    frame_idx: int,
    passes: tuple[InferredPass, ...],
    turnovers: tuple[InferredTurnover, ...],
    frame_rate: float,
    *,
    transformer=None,
) -> None:
    """In-flight pass highlights and turnover callouts for one frame."""
    _draw_pass_highlights(
        image,
        dets,
        frame_idx,
        passes,
        frame_rate,
        transformer=transformer,
    )
    for turnover in turnovers:
        _draw_turnover_notice(
            image,
            dets,
            turnover,
            frame_idx,
            frame_rate,
            transformer=transformer,
        )


def _render_pass_network(args, analysis: ClipAnalysis) -> None:
    """Render pass network overlay using a shared :class:`ClipAnalysis`."""
    from analytics.annotations import draw_pass_overlay
    from analytics.modes.pass_alternatives import PassEvent
    from analytics.passing import carrier_from_tracker_id, find_control_carrier

    metric = analysis.metric
    locks = analysis.locks("goal_distance")
    locked_goals = locks.locked_goal_defenders
    pass_by_frame = analysis.pass_by_frame
    scan = analysis.pass_scan
    network = build_pass_network(
        _clip_stem(analysis.source_video_path),
        list(scan.passes),
        list(scan.turnovers),
        metric=True,
    )
    fps = float(analysis.fps)
    stats_hold_frames = max(int(fps * 5), 1)
    events_by_frame = {e.frame_idx: e for e in network.passes}

    scorer = analysis.pass_scorer
    show_predictions = bool(getattr(args, "show_predictions", False))
    freeze_quality_threshold = float(getattr(args, "freeze_quality_threshold", 0.0))

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

            image = frame.copy()
            kps = metric.keypoints.get(frame_idx)
            transformer = metric.transforms.get(frame_idx)
            radar_transformer = metric.radar_transforms.get(frame_idx)
            image = annotate_players(image, dets, show_tracker_ids=True)
            image = annotate_ball(image, dets)
            carrier = find_control_carrier(dets, transformer=radar_transformer)
            if carrier is not None:
                feet = feet_xy(dets)
                draw_carrier_ground_ellipse(
                    image,
                    feet[carrier.index],
                    transformer=transformer,
                    color_bgr=CARRIER_SHADOW_BGR,
                    radius_m=0.5,
                    alpha=0.42,
                    filled=True,
                    thickness=1,
                )
            _draw_collaboration_web(image, dets, frame_idx, network.passes)
            draw_pass_network_frame_overlays(
                image,
                dets,
                frame_idx,
                network.passes,
                network.turnovers,
                fps,
                transformer=transformer,
            )

            if (
                show_predictions
                and frame_idx in events_by_frame
                and (qs := events_by_frame[frame_idx].quality_score) is not None
                and qs >= freeze_quality_threshold
            ):
                event = events_by_frame[frame_idx]
                freeze_carrier = carrier_from_tracker_id(dets, event.passer_tid)
                if freeze_carrier is not None:
                    options = scorer.top_options(frame_idx, dets, freeze_carrier, k=3)
                    if options:
                        freeze_event = PassEvent(
                            frame_idx=frame_idx,
                            carrier=freeze_carrier,
                            options=options,
                            top_score=options[0].score,
                        )
                        n_options = min(3, len(options))
                        reveal_frames = max(4, int(round(0.6 * fps)))
                        phases: list[tuple[int, int]] = [(0, reveal_frames)]
                        phases.extend((i, reveal_frames) for i in range(1, n_options + 1))
                        min_freeze = sum(h for _, h in phases)
                        extra_hold = max(0, int(round(2.5 * fps)) - min_freeze)
                        final_extra = max(4, int(round(1.0 * fps)))
                        if phases:
                            phases[-1] = (
                                phases[-1][0],
                                phases[-1][1] + extra_hold + final_extra,
                            )
                        for revealed, phase_hold in phases:
                            for step in range(phase_hold):
                                progress = (step + 1) / max(phase_hold, 1)
                                overlay = draw_pass_overlay(
                                    image,
                                    dets,
                                    freeze_event,
                                    revealed_options=revealed,
                                    reveal_progress=progress,
                                    weights=scorer._weights,
                                    metric=True,
                                    keypoints=kps,
                                    pitch_confidence=0.9,
                                    transformer=radar_transformer,
                                    show_lane_debug=kps is not None,
                                    show_radar=True,
                                    locked_goal_defenders=locked_goals,
                                )
                                sink.write_frame(overlay)

            image = draw_radar_minimap(
                image,
                dets,
                kps,
                pitch_confidence=0.9,
                locked_goal_defenders=locked_goals,
            )
            image = draw_hud_bar(image, f"PASS NETWORK  passes={network.n_passes}")
            sink.write_frame(draw_branding_tag(image))

        end_card = stats_end_card((width, height), network)
        for _ in range(stats_hold_frames):
            sink.write_frame(end_card)
    cap.release()
    print(f"Wrote {args.target_video_path}")


def run_pass_network(args, analysis: ClipAnalysis | None = None) -> None:
    """Render pass detection + collaboration overlay to ``args.target_video_path``."""
    if analysis is None:
        analysis = compute_clip_analysis(args, need_homography=True)
    _render_pass_network(args, analysis)

