"""Typed data-transfer objects shared between the RL and MPC modules."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobotState:
    """Minimal kinematic state of the robot.

    Attributes:
        x:     x-position in metres.
        y:     y-position in metres.
        theta: heading angle in radians.
        v:     linear velocity in m/s.
        w:     angular velocity in rad/s.
    """

    x: float
    y: float
    theta: float
    v: float = 0.0
    w: float = 0.0

    def as_array(self) -> np.ndarray:
        """Return ``[x, y, theta]`` as a 1-D NumPy array."""
        return np.array([self.x, self.y, self.theta], dtype=float)


@dataclass(frozen=True)
class LocalPlan:
    """A short-horizon position trajectory proposed by a planner.

    Attributes:
        positions: Array of shape ``(horizon, 2)`` with (x, y) waypoints.
        source:    Human-readable label: ``"original"``, ``"rl"``, or ``"filtered"``.
    """

    positions: np.ndarray  # shape: (horizon, 2)
    source: str  # "original" | "rl" | "filtered"

    def __post_init__(self) -> None:
        if self.positions.ndim != 2 or self.positions.shape[1] != 2:
            raise ValueError(
                f"positions must have shape (horizon, 2), got {self.positions.shape}"
            )


@dataclass(frozen=True)
class ControlCommand:
    """A unicycle control command produced by MPC or RL.

    Attributes:
        v: Linear velocity in m/s.
        w: Angular velocity in rad/s.
    """

    v: float
    w: float
