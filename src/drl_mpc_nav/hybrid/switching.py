"""Switching logic that decides whether to follow the RL reference or the
original MPC reference based on proximity to obstacles."""
from __future__ import annotations

from collections.abc import Sequence

from shapely.geometry import Point, Polygon


class HintSwitcher:
    """Activate/deactivate the RL-reference hint based on obstacle proximity.

    The switcher turns **on** when the original reference trajectory passes
    through an obstacle that is within *max_switch_distance* of the robot.
    It turns **off** after the robot has been at least *min_detach_distance*
    away from all obstacles for *min_detach_steps* consecutive steps.

    Args:
        max_switch_distance:  Robot-to-obstacle distance threshold for
                              switching on (metres).
        min_detach_distance:  Robot-to-obstacle distance threshold for
                              counting detach steps (metres).
        min_detach_steps:     Number of steps the robot must remain beyond
                              *min_detach_distance* before switching off.
    """

    def __init__(
        self,
        max_switch_distance: float,
        min_detach_distance: float,
        min_detach_steps: float = 5,
    ) -> None:
        self.switch_distance = max_switch_distance
        self.detach_distance = min_detach_distance
        self.detach_steps = min_detach_steps
        self.detach_cnt = 0
        self.switch_on = False

    def switch(
        self,
        current_position: Sequence[float],
        original_traj: Sequence[Sequence[float]],
        new_traj: Sequence[Sequence[float]],
        obstacle_list: Sequence[Sequence[Sequence[float]]],
    ) -> bool:
        """Evaluate whether the RL hint should be active this step.

        Args:
            current_position: Current (x, y) robot position.
            original_traj:    Waypoints of the MPC reference trajectory.
            new_traj:         Waypoints of the RL reference trajectory.
            obstacle_list:    List of polygon vertex lists (static + dynamic).

        Returns:
            ``True`` when the RL reference should be used; ``False`` otherwise.
        """
        cnt_flag = False
        for old_pos, _new_pos in zip(original_traj, new_traj, strict=False):
            for obstacle in obstacle_list:
                shapely_obstacle = Polygon(obstacle)
                dist = shapely_obstacle.distance(Point(current_position))
                if shapely_obstacle.contains(Point(old_pos[:2])):
                    if dist < self.switch_distance and not self.switch_on:
                        self.switch_on = True
                        return self.switch_on
                elif dist > self.detach_distance and self.switch_on:
                    if self.detach_cnt > self.detach_steps:
                        self.switch_on = False
                        self.detach_cnt = 0
                    elif not cnt_flag:
                        self.detach_cnt += 1
                        cnt_flag = True
        return self.switch_on
