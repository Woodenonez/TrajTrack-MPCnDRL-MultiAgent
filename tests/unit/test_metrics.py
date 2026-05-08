"""Unit tests for drl_mpc_nav.evaluation.metrics."""
import pytest

from drl_mpc_nav.evaluation.metrics import Metrics


def _dummy_trial(succeed: bool = True):
    """Create minimal valid inputs for one trial."""
    computation_time_list = [1.0, 2.0, 1.5, 1.8, 2.2]
    action_list = [(0.5, 0.1), (0.6, 0.2), (0.5, 0.1), (0.4, 0.0), (0.5, 0.1), (0.5, 0.1)]
    # Simple straight-line reference and actual trajectories
    ref_trajectory = [(float(i), 0.0) for i in range(10)]
    actual_trajectory = [(float(i), 0.1) for i in range(10)]
    # Square obstacle far from trajectory
    obstacle_list = [[(20.0, 20.0), (21.0, 20.0), (21.0, 21.0), (20.0, 21.0)]]
    return dict(
        computation_time_list=computation_time_list,
        succeed=succeed,
        action_list=action_list,
        ref_trajectory=ref_trajectory,
        actual_trajectory=actual_trajectory,
        obstacle_list=obstacle_list,
    )


class TestMetricsInit:
    def test_valid_mode(self):
        m = Metrics(mode="MPC")
        assert m.mode == "MPC"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="not recognised"):
            Metrics(mode="INVALID")

    def test_mode_case_insensitive(self):
        m = Metrics(mode="mpc")
        assert m.mode == "mpc"


class TestMetricsAddTrial:
    def test_single_successful_trial(self):
        m = Metrics(mode="MPC")
        m.add_trial_result(**_dummy_trial(succeed=True))
        assert m.success_rate == pytest.approx(1.0)
        assert len(m.trial_list) == 1

    def test_single_failed_trial(self):
        m = Metrics(mode="MPC")
        m.add_trial_result(**_dummy_trial(succeed=False))
        assert m.success_rate == pytest.approx(0.0)

    def test_mixed_trials(self):
        m = Metrics(mode="MPC")
        m.add_trial_result(**_dummy_trial(succeed=True))
        m.add_trial_result(**_dummy_trial(succeed=False))
        assert m.success_rate == pytest.approx(0.5)


class TestMetricsGetAverage:
    def test_keys_present(self):
        m = Metrics(mode="MPC")
        m.add_trial_result(**_dummy_trial())
        avg = m.get_average()
        for key in ("computation_time", "deviation_distance", "smoothness",
                    "clearance", "finish_time", "success_rate"):
            assert key in avg

    def test_computation_time_three_values(self):
        m = Metrics(mode="MPC")
        m.add_trial_result(**_dummy_trial())
        avg = m.get_average()
        assert len(avg["computation_time"]) == 3

    def test_failed_trial_finish_time_negative(self):
        m = Metrics(mode="MPC")
        m.add_trial_result(**_dummy_trial(succeed=False))
        avg = m.get_average()
        assert avg["finish_time"] == pytest.approx(-1)

    def test_success_rate_in_average(self):
        m = Metrics(mode="MPC")
        for _ in range(3):
            m.add_trial_result(**_dummy_trial(succeed=True))
        m.add_trial_result(**_dummy_trial(succeed=False))
        avg = m.get_average()
        assert avg["success_rate"] == pytest.approx(0.75)


class TestMetricsWriteLatex:
    def test_returns_string(self):
        m = Metrics(mode="MPC")
        m.add_trial_result(**_dummy_trial())
        latex = m.write_latex()
        assert isinstance(latex, str)
        assert "MPC" in latex
