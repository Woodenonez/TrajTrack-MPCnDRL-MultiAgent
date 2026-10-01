"""Regressions for shared-map validation and bounded MPC completion."""
from contextlib import redirect_stdout
import importlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests/demos")]


class EpisodeStartupTests(unittest.TestCase):
    def test_checker_cannot_mutate_the_experiment_map(self):
        from helper import generate_map

        def moving_check(env):
            env.reset()
            env.unwrapped.agent.position = np.array([-99.0, -99.0])

        for name, algorithm in (("test_ddpg", "PerDDPG"), ("test_dqn", "DQN"), ("evaluation", "PerDDPG")):
            with self.subTest(loader=name):
                module = importlib.import_module(name)
                factory = generate_map(1, 1, 2)
                calls = 0

                def bounded_factory():
                    nonlocal calls
                    calls += 1
                    if calls > 8:
                        raise AssertionError("reset repeatedly retried a checker-mutated map")
                    return factory()

                policy = SimpleNamespace(policy=SimpleNamespace(load_state_dict=lambda _: None))
                with patch.object(module.env_checker, "check_env", moving_check), \
                     patch.object(module, algorithm, return_value=policy), \
                     patch.object(module.torch, "load", return_value={}):
                    _, env = module.load_rl_model_env(bounded_factory, 1)
                try:
                    np.testing.assert_array_equal(factory()[0].position, [0.6, 3.5])
                    np.testing.assert_array_equal(env.unwrapped.agent.position, [0.6, 3.5])
                finally:
                    env.close()

    def test_mpc_completion_ends_outer_episode_without_a_restart(self):
        from util.run_records import EpisodeRecord

        class Env:
            def __init__(self):
                self.unwrapped = self
                self.agent = SimpleNamespace(position=np.zeros(2), angle=0.0)
                self.goal = SimpleNamespace(position=np.ones(2))
                self.path = SimpleNamespace(coords=[(0.0, 0.0), (1.0, 1.0)])
                self.obstacles = []
                self.speeds = self.angular_velocities = [0.0]
                self.traversed_positions = [(0.0, 0.0)]
                self.collided = False
                self.resets = 0

            def get_map_description(self):
                return ()

            def reset(self):
                self.resets += 1
                if self.resets > 1:
                    raise AssertionError("MPC completion restarted the episode")
                return {}, {}

            def set_agent_state(self, *args):
                pass

            def step(self, action):
                return {}, 0.0, False, False, {"success": False}

        class CompletedSolver:
            def __init__(self, *args, **kwargs):
                self.state = np.zeros(3)
                self.last_action = np.zeros(2)
                self.ref_traj = np.zeros((1, 3))

            def update_static_constraints(self, *args):
                pass

            def load_init_state(self, *args):
                pass

            def set_work_mode(self, *args, **kwargs):
                pass

            def set_ref_trajectory(self, *args):
                pass

            def get_local_ref_traj(self):
                return (self.ref_traj,)

            def get_action(self, *args):
                return None

        for name in ("test_ddpg", "test_dqn", "evaluation"):
            with self.subTest(workflow=name):
                module = importlib.import_module(name)
                env, record = Env(), EpisodeRecord()
                output = io.StringIO()
                with patch.object(module, "load_rl_model_env", return_value=(None, env)), \
                     patch.object(module, "generate_map", return_value=None), \
                     patch.object(module.MPCConfig, "from_yaml", return_value=SimpleNamespace()), \
                     patch.object(module, "TrajectoryGenerator", CompletedSolver), \
                     patch.object(module, "get_geometric_map", return_value=SimpleNamespace(processed_obstacle_list=[], obstacle_list=[])), \
                     redirect_stdout(output):
                    run = module.main_process if name == "evaluation" else module.main
                    run(decision_mode=0, max_steps=1, record=record)
                self.assertEqual(record.termination, "completed")
                self.assertFalse(record.budget_exhausted)
                self.assertNotIn("nan", output.getvalue())


if __name__ == "__main__":
    unittest.main()
