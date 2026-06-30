# analytics package — player-motion analytics for the sports soccer example

import sys
from pathlib import Path

# The `sports` library is not pip-installed in this checkout; it lives at the
# repository root (…/sports/sports/). Ensure that root is importable no matter
# what the current working directory is when running analytics/main.py.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if (_REPO_ROOT / "sports").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analytics.modes.direction import run_direction
from analytics.modes.distance import run_distance
from analytics.modes.pass_alternatives import run_pass_alternatives
from analytics.modes.pass_network import run_pass_network
from analytics.modes.speed import run_speed
from analytics.modes.speed_and_distance import run_speed_and_distance
from analytics.run_all import run_all

__all__ = [
    "run_all",
    "run_direction",
    "run_distance",
    "run_pass_alternatives",
    "run_pass_network",
    "run_speed",
    "run_speed_and_distance",
]
