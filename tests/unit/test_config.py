"""Unit tests for drl_mpc_nav config dataclasses."""
from pathlib import Path

import numpy as np
import pytest

from drl_mpc_nav.config import DecisionMode, EvalConfig, TrainConfig
from drl_mpc_nav.config.schema import ControlCommand, LocalPlan, RobotState


class TestDecisionMode:
    def test_values(self):
        assert DecisionMode.MPC.value == "mpc"
        assert DecisionMode.DDPG.value == "ddpg"
        assert DecisionMode.HYBRID.value == "hybrid"

    def test_from_string(self):
        assert DecisionMode("hybrid") is DecisionMode.HYBRID


class TestEvalConfig:
    def test_defaults(self):
        cfg = EvalConfig()
        assert cfg.rl_index == 1
        assert cfg.decision_mode == DecisionMode.HYBRID
        assert cfg.max_steps == 200
        assert cfg.seed == 0

    def test_custom_values(self):
        cfg = EvalConfig(
            rl_index=0,
            decision_mode=DecisionMode.MPC,
            max_steps=50,
            scene=(2, 1, 1),
        )
        assert cfg.rl_index == 0
        assert cfg.decision_mode == DecisionMode.MPC
        assert cfg.max_steps == 50
        assert cfg.scene == (2, 1, 1)


class TestTrainConfig:
    def test_defaults(self):
        cfg = TrainConfig()
        assert cfg.total_timesteps == 100_000
        assert cfg.n_envs == 4
        assert not cfg.load_checkpoint

    def test_output_dir_is_path(self):
        cfg = TrainConfig(output_dir=Path("my/dir"))
        assert isinstance(cfg.output_dir, Path)


class TestRobotState:
    def test_as_array(self):
        state = RobotState(x=1.0, y=2.0, theta=0.5)
        arr = state.as_array()
        assert arr.shape == (3,)
        np.testing.assert_array_equal(arr, [1.0, 2.0, 0.5])

    def test_frozen(self):
        state = RobotState(x=0.0, y=0.0, theta=0.0)
        with pytest.raises((AttributeError, TypeError)):
            state.x = 1.0  # type: ignore[misc]


class TestLocalPlan:
    def test_valid_creation(self):
        positions = np.zeros((10, 2))
        plan = LocalPlan(positions=positions, source="rl")
        assert plan.source == "rl"

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            LocalPlan(positions=np.zeros((10, 3)), source="rl")


class TestControlCommand:
    def test_creation(self):
        cmd = ControlCommand(v=1.0, w=0.1)
        assert cmd.v == pytest.approx(1.0)
        assert cmd.w == pytest.approx(0.1)
