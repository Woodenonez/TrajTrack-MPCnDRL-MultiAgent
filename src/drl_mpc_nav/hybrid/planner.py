"""Dynamic-obstacle position estimation and geometry helpers used by the
hybrid planner."""
from __future__ import annotations

from collections.abc import Sequence


def est_dyn_obs_positions(
    last_pos: Sequence[float],
    current_pos: Sequence[float],
    steps: int = 20,
    obs_size: float = 1.6,
) -> list[list[float]]:
    """Predict future positions of a dynamic obstacle using constant velocity.

    Args:
        last_pos:    (x, y) position at the previous time step.
        current_pos: (x, y) position at the current time step.
        steps:       Number of future steps to predict.
        obs_size:    Side length (m) of the square bounding box used to
                     represent the obstacle for the MPC constraint.

    Returns:
        List of ``steps`` entries, each a 6-element list
        ``[x, y, width, height, angle, flag]`` compatible with the MPC
        dynamic-obstacle constraint format.
    """
    d_pos = [current_pos[0] - last_pos[0], current_pos[1] - last_pos[1]]
    return [
        [
            current_pos[0] + d_pos[0] * (i + 1),
            current_pos[1] + d_pos[1] * (i + 1),
            obs_size,
            obs_size,
            0,
            1,
        ]
        for i in range(steps)
    ]


def circle_to_rect(
    pos: Sequence[float],
    radius: float = 1.6,
) -> list[list[float]]:
    """Return the four corners of an axis-aligned square centred at *pos*.

    Args:
        pos:    (x, y) centre of the circle.
        radius: Half-side-length of the equivalent square (= circle radius).

    Returns:
        List of four ``[x, y]`` corners in counter-clockwise order.
    """
    x, y = pos[0], pos[1]
    return [
        [x - radius, y - radius],
        [x + radius, y - radius],
        [x + radius, y + radius],
        [x - radius, y + radius],
    ]
