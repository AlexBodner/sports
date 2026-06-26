"""Detection class ids — matches ``examples/soccer/main.py`` and Inference football models."""

import numpy as np
import supervision as sv

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

# Team ids (stabilized jersey clustering).
TEAM_NONE = -1
TEAM_LEFT = 0
TEAM_RIGHT = 1

# Team colors — match ``main.py`` COLORS[0] / COLORS[1].
TEAM_COLORS = [
    sv.Color.from_hex("#FF1493"),
    sv.Color.from_hex("#00BFFF"),
]
REFEREE_COLOR = sv.Color.from_hex("#FFD700")
NEUTRAL_COLOR = sv.Color.from_hex("#CCCCCC")
TEAM_VIS_PALETTE = sv.ColorPalette([*TEAM_COLORS, REFEREE_COLOR])

# Role aliases used by possession and pass modules.
ROLE_BALL = BALL_CLASS_ID
ROLE_GOALKEEPER = GOALKEEPER_CLASS_ID
ROLE_PLAYER = PLAYER_CLASS_ID
ROLE_REFEREE = REFEREE_CLASS_ID


def team_vis_class_ids(teams: np.ndarray) -> np.ndarray:
    """Map stabilized team ids to supervision ``ColorLookup.CLASS`` indices."""
    return np.where(np.isin(teams, (0, 1)), teams, 2).astype(int)
