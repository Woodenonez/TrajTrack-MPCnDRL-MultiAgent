"""Unit tests for drl_mpc_nav.hybrid.switching."""

from drl_mpc_nav.hybrid.switching import HintSwitcher


def _square(cx: float, cy: float, half: float = 0.5):
    """Return a simple square polygon as a list of (x, y) vertices."""
    return [
        [cx - half, cy - half],
        [cx + half, cy - half],
        [cx + half, cy + half],
        [cx - half, cy + half],
    ]


class TestHintSwitcher:
    def test_starts_off(self):
        hs = HintSwitcher(max_switch_distance=5.0, min_detach_distance=2.0)
        assert hs.switch_on is False

    def test_no_obstacles_stays_off(self):
        hs = HintSwitcher(max_switch_distance=5.0, min_detach_distance=2.0)
        result = hs.switch(
            current_position=[0.0, 0.0],
            original_traj=[[1.0, 0.0], [2.0, 0.0]],
            new_traj=[[1.0, 0.5], [2.0, 0.5]],
            obstacle_list=[],
        )
        assert result is False

    def test_switches_on_when_ref_passes_through_obstacle(self):
        hs = HintSwitcher(max_switch_distance=5.0, min_detach_distance=2.0)
        # Obstacle at (1, 0); original traj passes through it; robot is nearby
        obstacle = _square(1.0, 0.0, half=0.6)
        result = hs.switch(
            current_position=[0.0, 0.0],
            original_traj=[[1.0, 0.0]],   # inside obstacle
            new_traj=[[1.0, 1.0]],
            obstacle_list=[obstacle],
        )
        assert result is True

    def test_remains_on_until_detach_condition_met(self):
        hs = HintSwitcher(
            max_switch_distance=5.0,
            min_detach_distance=1.0,
            min_detach_steps=2,
        )
        obstacle = _square(1.0, 0.0, half=0.6)
        # First call: switch on
        hs.switch(
            current_position=[0.0, 0.0],
            original_traj=[[1.0, 0.0]],
            new_traj=[[1.0, 1.0]],
            obstacle_list=[obstacle],
        )
        assert hs.switch_on is True
        # Robot still near obstacle — should remain on
        result = hs.switch(
            current_position=[0.5, 0.0],
            original_traj=[[1.0, 0.0]],
            new_traj=[[1.0, 1.0]],
            obstacle_list=[obstacle],
        )
        assert result is True
