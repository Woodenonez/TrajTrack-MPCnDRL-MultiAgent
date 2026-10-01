"""Unit tests for geometry helpers in drl_mpc_nav.hybrid.planner."""
import pytest

from drl_mpc_nav.hybrid.planner import circle_to_rect, est_dyn_obs_positions


class TestCircleToRect:
    def test_returns_four_corners(self):
        corners = circle_to_rect([0.0, 0.0], radius=1.0)
        assert len(corners) == 4

    def test_correct_extents_unit_radius(self):
        corners = circle_to_rect([0.0, 0.0], radius=1.0)
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        assert min(xs) == pytest.approx(-1.0)
        assert max(xs) == pytest.approx(1.0)
        assert min(ys) == pytest.approx(-1.0)
        assert max(ys) == pytest.approx(1.0)

    def test_centred_at_non_origin(self):
        cx, cy, r = 3.0, -2.0, 0.5
        corners = circle_to_rect([cx, cy], radius=r)
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        assert min(xs) == pytest.approx(cx - r)
        assert max(xs) == pytest.approx(cx + r)
        assert min(ys) == pytest.approx(cy - r)
        assert max(ys) == pytest.approx(cy + r)

    def test_default_radius(self):
        """Default radius should be 1.6 (= DYN_OBS_SIZE = 0.8 + 0.8)."""
        corners = circle_to_rect([0.0, 0.0])
        xs = [c[0] for c in corners]
        assert min(xs) == pytest.approx(-1.6)
        assert max(xs) == pytest.approx(1.6)


class TestEstDynObsPositions:
    def test_output_length(self):
        last = [0.0, 0.0]
        cur  = [1.0, 0.0]
        result = est_dyn_obs_positions(last, cur, steps=10)
        assert len(result) == 10

    def test_each_entry_has_six_fields(self):
        result = est_dyn_obs_positions([0.0, 0.0], [1.0, 0.0], steps=5)
        for entry in result:
            assert len(entry) == 6

    def test_constant_velocity_prediction(self):
        """Each step should advance by the same delta as the observed step."""
        last = [0.0, 0.0]
        cur  = [1.0, 2.0]
        result = est_dyn_obs_positions(last, cur, steps=3)
        # step 1: [2, 4], step 2: [3, 6], step 3: [4, 8]
        assert result[0][0] == pytest.approx(2.0)
        assert result[0][1] == pytest.approx(4.0)
        assert result[1][0] == pytest.approx(3.0)
        assert result[2][1] == pytest.approx(8.0)

    def test_stationary_obstacle(self):
        """A stationary obstacle should predict the same position every step."""
        pos = [5.0, 3.0]
        result = est_dyn_obs_positions(pos, pos, steps=5)
        for entry in result:
            assert entry[0] == pytest.approx(5.0)
            assert entry[1] == pytest.approx(3.0)

    def test_obs_size_propagated(self):
        result = est_dyn_obs_positions([0.0, 0.0], [1.0, 0.0], steps=1, obs_size=2.0)
        assert result[0][2] == pytest.approx(2.0)
        assert result[0][3] == pytest.approx(2.0)
