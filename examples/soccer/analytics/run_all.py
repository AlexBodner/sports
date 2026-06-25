"""analytics/run_all.py — in-process orchestrator for all analytics renders.

Computes the shared :class:`~analytics.clip_pipeline.ClipAnalysis` ONCE (one BoTSORT pass,
one team-classifier fit, one set of homography maps, one kinematics integration, one pass
scan) and then drives each mode renderer in-process. Each mode still writes its own output
video; only the expensive shared groundwork is reused.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

from analytics.clip_pipeline import ClipAnalysis, compute_clip_analysis
from analytics.modes.direction import run_direction
from analytics.modes.distance import run_distance
from analytics.modes.pass_alternatives import run_pass_alternatives
from analytics.modes.pass_network import run_pass_network
from analytics.modes.speed_and_distance import run_speed_and_distance
from analytics.modes.speed import run_speed

# (output suffix, human label, dispatch key) for renders in run order.
_RENDER_PLAN = (
    ("direction", "DIRECTION", "direction"),
    ("speed", "SPEED", "speed"),
    ("distance", "DISTANCE", "distance"),
    ("speed-distance-all", "SPEED_AND_DISTANCE (all players)", "speed-distance-all"),
    ("speed-distance-single", "SPEED_AND_DISTANCE (spotlight)", "speed-distance-single"),
    ("pass-network", "PASS_NETWORK", "pass-network"),
    ("pass-alternatives", "PASS_ALTERNATIVES", "pass-alternatives"),
)


def _derive_target(base_path: str, suffix: str) -> str:
    """``data/renders/08fd33_0.mp4`` + ``direction`` → ``…/08fd33_0-direction.mp4``."""
    p = Path(base_path)
    return str(p.with_name(f"{p.stem}-{suffix}{p.suffix or '.mp4'}"))


def _mode_args(args, *, target: str, track_id: int | None = None):
    """Shallow copy of ``args`` with a per-mode target path and spotlight track id."""
    new = copy.copy(args)
    new.target_video_path = target
    new.track_id = track_id
    return new


def _pick_spotlight_track_id(analysis: ClipAnalysis) -> int | None:
    """Pick the most-active tracked player (max cumulative distance) for the spotlight.

    Deterministic from the shared kinematics; an explicit ``--track-id`` overrides it.
    """
    best_tid: int | None = None
    best_distance = -1.0
    for tid, track in analysis.tracks.items():
        distance = float(getattr(track, "distance_m", 0.0) or 0.0)
        if distance > best_distance:
            best_distance = distance
            best_tid = int(tid)
    return best_tid


def run_all(args) -> list[str]:
    """Compute the shared analysis once and render all seven analytics videos in-process."""
    t_start = time.time()

    print("── run-all: computing shared ClipAnalysis ─────────")
    analysis = compute_clip_analysis(args, need_homography=True)
    # Warm pass scan once so PASS_NETWORK / PASS_ALTERNATIVES reuse the same result.
    n_passes = len(analysis.pass_scan.passes)
    print(
        f"Shared analysis ready in {time.time() - t_start:.1f}s "
        f"(pass scan: {n_passes} passes)."
    )

    base = args.target_video_path
    explicit_spotlight = getattr(args, "track_id", None)
    spotlight_id = explicit_spotlight if explicit_spotlight is not None else _pick_spotlight_track_id(analysis)
    print(f"Spotlight (SPEED_AND_DISTANCE) track id: {spotlight_id}")

    outputs: list[str] = []
    timings: list[tuple[str, float]] = []
    for suffix, label, key in _RENDER_PLAN:
        target = _derive_target(base, suffix)
        track_id = spotlight_id if suffix == "speed-distance-single" else None
        t_mode = time.time()
        print(f"── run-all: rendering {label} → {target} ─────────")
        mode_args = _mode_args(args, target=target, track_id=track_id)
        if key == "direction":
            run_direction(mode_args, analysis)
        elif key == "speed":
            run_speed(mode_args, analysis)
        elif key == "distance":
            run_distance(mode_args, analysis)
        elif key.startswith("speed-distance"):
            run_speed_and_distance(mode_args, analysis)
        elif key == "pass-network":
            run_pass_network(mode_args, analysis)
        elif key == "pass-alternatives":
            run_pass_alternatives(mode_args, analysis)
        else:
            raise ValueError(f"Unknown render key: {key}")
        outputs.append(target)
        timings.append((label, time.time() - t_mode))

    total_time = time.time() - t_start
    print("── run-all: done ─────────────────────────────────────────────────────")
    for label, dt in timings:
        print(f"  {label:<28} {dt:6.1f}s")
    print(f"  {'TOTAL':<28} {total_time:6.1f}s")
    print(f"  All {len(outputs)} renders used a single shared tracking pass.")
    print("Outputs:")
    for path in outputs:
        print(f"  {path}")
    return outputs
