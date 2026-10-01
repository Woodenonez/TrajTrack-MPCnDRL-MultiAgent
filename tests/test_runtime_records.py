"""Runtime boundaries that must reject misleading experiment evidence."""

from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tests/demos"), str(ROOT / "src")]

import run_experiment
smoke_spec = importlib.util.spec_from_file_location("runtime_smoke_script", ROOT / "scripts/smoke.py")
smoke = importlib.util.module_from_spec(smoke_spec)
sys.modules["runtime_smoke_script"] = smoke
smoke_spec.loader.exec_module(smoke)


class NativeBindingTests(unittest.TestCase):
    def test_preflight_rejects_binding_shadowed_on_import_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            optimizer = root / "solver/navi_default"
            optimizer.mkdir(parents=True)
            (optimizer / "navi_default.so").touch()
            config = root / "config.yaml"
            config.write_text("optimizer_name: navi_default\nbuild_directory: solver\n")
            shadow = root / "shadow"
            shadow.mkdir()
            (shadow / "navi_default.py").write_text("solver = object\n")
            spec = run_experiment.RunSpec("mpc-demo", 0, None, None, 4, False, 1, None, config, root / "solver")
            with patch.object(sys, "path", [str(shadow), *sys.path]):
                with self.assertRaisesRegex(RuntimeError, "binding|import"):
                    run_experiment._prepare(run_experiment.ExperimentArgs(workflow="mpc-demo"), [spec])

    def test_preflight_selects_only_compatible_python_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            optimizer = root / "navi_default"
            optimizer.mkdir()
            (optimizer / "navi_default.cpython-310-x86_64-linux-gnu.so").touch()
            compatible = optimizer / "navi_default.so"
            compatible.touch()
            spec = run_experiment.RunSpec("mpc-demo", 0, None, None, 4, False, 1, None, root / "config.yaml", root)
            self.assertEqual(run_experiment._native_artifact(spec, "navi_default", root), compatible)
            compatible.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "binding"):
                run_experiment._native_artifact(spec, "navi_default", root)


class LauncherStatusTests(unittest.TestCase):
    def test_solver_error_and_empty_demo_trace_fail_the_launcher(self) -> None:
        spec = run_experiment.RunSpec("ddpg-demo", 2, 0, (1, 3, 2), None, False, 1,
                                      Path("/tmp/model.pt"), Path("/tmp/config.yaml"), Path("/tmp/solver"))
        for termination, samples in (("solver_error", 1), ("unknown", 0)):
            with self.subTest(termination=termination):
                demo = ModuleType("test_ddpg")

                def main(**kwargs):
                    record = kwargs["record"]
                    if record is not None:
                        record.termination = termination
                        record.timing_ms.extend([1.0] * samples)
                    return [1.0] * samples

                demo.main = main
                with patch.object(run_experiment, "resolve_runs", return_value=[spec]), \
                     patch.object(run_experiment, "_prepare", return_value=(None, Path("/tmp/solver"))), \
                     patch.object(run_experiment, "_add_import_paths"), \
                     patch.dict(sys.modules, {"test_ddpg": demo}), \
                     redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "solver|empty|trace"):
                        run_experiment.run_experiment(run_experiment.ExperimentArgs())


class SmokeStatusTests(unittest.TestCase):
    def test_smoke_rejects_failed_child_record(self) -> None:
        class Policy:
            def predict(self, observation, deterministic=True):
                return np.array([0.0]), None

        class Environment:
            def reset(self, seed=None):
                return {"internal": np.array([0.0])}, {}

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoints = root / "checkpoints"
            for algorithm in ("ddpg", "dqn"):
                for modality in ("image", "ray"):
                    path = checkpoints / algorithm / modality / "best_model.pt"
                    path.parent.mkdir(parents=True)
                    path.write_bytes(b"fixture")
            demo = ModuleType("test_ddpg")
            demo.generate_map = lambda *args: None
            demo.load_rl_model_env = lambda *args, **kwargs: (Policy(), Environment())
            with patch.dict(sys.modules, {"test_ddpg": demo, "test_dqn": demo}), \
                 patch.object(run_experiment, "_add_import_paths"), \
                 patch.object(run_experiment, "_seed_explicitly"), \
                 patch.object(run_experiment, "run_experiment", side_effect=lambda args: (
                     args.output_dir.mkdir(), np.savez_compressed(args.output_dir / "trial_001_failed.npz",
                                                                  termination="solver_error"))), \
                 redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "solver_error"):
                    smoke.run_smoke(root / "run", checkpoints, root / "solver", steps=1)


class FleetStatusTests(unittest.TestCase):
    def test_prior_completion_remains_completed_at_later_budget_exit(self) -> None:
        from _scenario_simulator import Simulator

        class Generator:
            def __init__(self, completes_first: bool) -> None:
                self.completes_first = completes_first
                self.checks = 0
                self.state = np.zeros(3)
                self.past_actions: list[np.ndarray] = []
                self.final_goal = np.ones(3)

            def load_init_state(self, *args):
                pass

            def set_work_mode(self, *args):
                pass

            def set_ref_trajectory(self, *args):
                pass

            def get_local_ref_traj(self):
                return np.zeros((1, 3)),

            def run_step(self, *args, **kwargs):
                action = np.zeros(2)
                self.past_actions.append(action)
                return [action], [self.state.copy()], 0.0

            def check_termination_condition(self, *args):
                self.checks += 1
                return self.completes_first and self.checks == 1

        simulator = Simulator.__new__(Simulator)
        simulator.config = SimpleNamespace(Nstcobs=0, nstcobs=12, Ndynobs=0, ndynobs=6, action_steps=1)
        simulator.N_hor = 1
        simulator.ts = 0.2
        simulator.use_tcp = False
        simulator.get_other_robot_states = lambda robot_id: []
        simulator.robot_dict = {
            index: SimpleNamespace(start=np.zeros(3), end=np.ones(3), mode="work", ref_path=[], color="b",
                                   traj_gen=Generator(index == 0), done=False, termination="unknown", budget_exhausted=False)
            for index in (0, 1)
        }
        scanner = SimpleNamespace(get_full_obstacle_list=lambda **kwargs: [])
        with redirect_stdout(io.StringIO()):
            result = simulator.run(lambda: (None, []), scanner, plot_in_loop=False, max_steps=2)
        self.assertTrue(result[0].done)
        self.assertEqual(result[0].termination, "completed")
        self.assertFalse(result[0].budget_exhausted)
        self.assertEqual(result[1].termination, "step_limit")
        self.assertTrue(result[1].budget_exhausted)


if __name__ == "__main__":
    unittest.main()
