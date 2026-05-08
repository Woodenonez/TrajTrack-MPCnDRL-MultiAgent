"""Evaluation metrics collected over one or more navigation trials."""
from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

import numpy as np
from shapely.geometry import Point, Polygon


class Metrics:
    """Accumulates per-trial measurements and computes aggregate statistics.

    Tracked quantities per trial:

    1. **Computation time** — mean, max, variance across all steps.
    2. **Deviation distance** — mean and max distance from the reference path.
    3. **Action smoothness** — second-order finite differences of v and ω.
    4. **Minimal clearance** — smallest obstacle distance along the trajectory.
    5. **Finish time** — number of steps taken (−1 if the episode failed).
    6. **Success rate** — fraction of trials that reached the goal.

    Args:
        mode: Label for the planning mode (e.g. ``"MPC"``, ``"DDPG-V"``).

    Raises:
        ValueError: If *mode* is not in the recognised list.
    """

    _mode_list = [
        "MPC",
        "DQN-L",
        "DQN-V",
        "HYB-DQN-L",
        "HYB-DQN-V",
        "DDPG-L",
        "DDPG-V",
        "HYB-DDPG-L",
        "HYB-DDPG-V",
    ]

    def __init__(self, mode: str) -> None:
        if mode.upper() not in self._mode_list:
            raise ValueError(
                f"Mode '{mode.upper()}' not recognised (should be one of {self._mode_list})."
            )
        self.mode = mode
        self.trial_list: list[dict] = []
        self.success_rate: float = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_trial_result(
        self,
        computation_time_list: Sequence[float],
        succeed: bool,
        action_list: Sequence[tuple[float, float]],
        ref_trajectory: Sequence[tuple[float, float]],
        actual_trajectory: Sequence[tuple[float, float]],
        obstacle_list: Sequence[Sequence[Sequence[float]]],
    ) -> None:
        """Record measurements for one completed trial.

        Args:
            computation_time_list: Per-step wall-clock times in milliseconds.
            succeed:               Whether the robot reached the goal.
            action_list:           Per-step ``(v, ω)`` commands.
            ref_trajectory:        Global reference path waypoints.
            actual_trajectory:     Positions visited by the robot.
            obstacle_list:         Static-obstacle polygon vertex lists.
        """
        metric_dict: dict = {}
        metric_dict["computation_time"] = self._get_computation_time(
            list(computation_time_list)
        )
        metric_dict["deviation_distance"] = self._get_deviation_distance(
            list(ref_trajectory), list(actual_trajectory)
        )
        metric_dict["smoothness"] = self._get_smoothness(list(action_list))
        metric_dict["clearance"] = self._get_minimal_obstacle_distance(
            list(actual_trajectory), list(obstacle_list)
        )
        metric_dict["finish_time"] = self._get_finish_time_steps(
            list(computation_time_list), succeed=succeed
        )
        metric_dict["success"] = metric_dict["finish_time"] > 0
        self.trial_list.append(metric_dict)
        self._update_success_rate()

    def get_average(self, round_digit: int = 4) -> dict:
        """Return a dict of averaged metrics across all recorded trials.

        Args:
            round_digit: Number of decimal places to round to.

        Returns:
            Dictionary with keys ``computation_time``, ``deviation_distance``,
            ``smoothness``, ``clearance``, ``finish_time``, ``success_rate``.
        """
        all_ct: list = []
        all_dd: list = []
        all_sm: list = []
        all_cl: list = []
        all_ft: list = []

        for trial in self.trial_list:
            all_ct.append(trial["computation_time"])
            all_dd.append(trial["deviation_distance"])
            all_sm.append(trial["smoothness"])
            all_cl.append(trial["clearance"])
            if trial["success"]:
                all_ft.append(trial["finish_time"])

        if len(all_ct) > 10:
            all_ct = all_ct[5:]  # discard warm-up trials for stability

        if not all_ft:
            all_ft = [-1]

        avg: dict = {}
        avg["computation_time"] = [
            round(statistics.mean([x[0] for x in all_ct]), round_digit),
            round(statistics.mean([x[1] for x in all_ct]), round_digit),
            round(statistics.mean([x[2] for x in all_ct]), round_digit),
        ]
        avg["deviation_distance"] = [
            round(statistics.mean([x[0] for x in all_dd]), round_digit),
            round(statistics.mean([x[1] for x in all_dd]), round_digit),
        ]
        avg["smoothness"] = [
            round(float(statistics.mean([x[0] for x in all_sm])), round_digit),
            round(float(statistics.mean([x[1] for x in all_sm])), round_digit),
        ]
        avg["clearance"] = round(statistics.mean(all_cl), round_digit)
        avg["finish_time"] = round(statistics.mean(all_ft), round_digit)
        avg["success_rate"] = self.success_rate
        return avg

    def write_latex(self, round_digit: int = 4) -> str:
        """Return a LaTeX table row summarising the averaged metrics.

        Args:
            round_digit: Number of decimal places to round to.

        Returns:
            A string fragment suitable for pasting into a LaTeX table.
        """
        data = self.get_average(round_digit)
        ct = data["computation_time"]
        dd = data["deviation_distance"]
        sm = data["smoothness"]
        return (
            f"&  & {self.mode} "
            f"& {ct[0]} & {ct[1]} & {ct[2]} "
            f"& {dd[0]} & {dd[1]} "
            f"& {sm[0]} & {sm[1]} "
            f"& {int(data['finish_time'])} "
            f"& {int(data['success_rate'] * 100)} \\\\ % OK"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_computation_time(
        self, computation_time_list: list[float]
    ) -> list[float]:
        return [
            statistics.mean(computation_time_list),
            max(computation_time_list),
            statistics.variance(computation_time_list),
        ]

    def _get_deviation_distance(
        self,
        ref_traj: list,
        actual_traj: list,
    ) -> list[float]:
        deviation_dists = [
            min(
                math.hypot(ref_pos[0] - pos[0], ref_pos[1] - pos[1])
                for ref_pos in ref_traj
            )
            for pos in actual_traj
        ]
        return [statistics.mean(deviation_dists), max(deviation_dists)]

    def _get_smoothness(self, action_list: list) -> list[float]:
        speeds = np.array(action_list)[:, 0]
        angular_speeds = np.array(action_list)[:, 1]
        return [
            float(statistics.mean(np.abs(np.diff(speeds, n=2)))),
            float(statistics.mean(np.abs(np.diff(angular_speeds, n=2)))),
        ]

    def _get_minimal_obstacle_distance(
        self,
        trajectory: list,
        obstacles: list,
    ) -> float:
        dist_list = [
            min(Polygon(obs).distance(Point(pos)) for obs in obstacles)
            for pos in trajectory
        ]
        return min(dist_list)

    def _get_finish_time_steps(
        self,
        computation_time_list: list[float],
        succeed: bool,
    ) -> int:
        return len(computation_time_list) if succeed else -1

    def _update_success_rate(self) -> None:
        self.success_rate = sum(t["success"] for t in self.trial_list) / len(
            self.trial_list
        )
