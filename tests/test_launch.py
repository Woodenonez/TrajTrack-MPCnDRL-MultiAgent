"""Grouped interface checks for the small experiment and solver CLIs."""

from __future__ import annotations

import os
import contextlib
import importlib.util
import io
from unittest import mock
import numpy as np
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests/demos"))

import build_solver
import run_experiment


class ExperimentResolutionTests(unittest.TestCase):
    def test_entry_defaults_preserve_their_distinct_presets(self) -> None:
        defaults = (
            ("ddpg-demo", [(2, 0, (1, 3, 2), None, True)]),
            ("dqn-demo", [(2, 0, (1, 1, 2), None, True)]),
            ("mpc-demo", [(0, None, None, 4, True)]),
            (
                "ddpg-eval",
                [
                    (0, 1, (1, 3, 2), None, False),
                    (1, 0, (1, 3, 2), None, False),
                    (2, 0, (1, 3, 2), None, False),
                ],
            ),
        )
        for workflow, expected in defaults:
            with self.subTest(workflow=workflow):
                runs = run_experiment.resolve_runs(run_experiment.ExperimentArgs(workflow=workflow))
                self.assertEqual(
                    [(r.decision_mode, r.rl_index, r.scene_option, r.case_index, r.to_plot) for r in runs],
                    expected,
                )
                self.assertTrue(all(r.mpc_config.is_absolute() for r in runs))
                self.assertTrue(all(r.max_steps is None for r in runs))
                self.assertTrue(all(r.checkpoint is None if workflow == "mpc-demo" else r.checkpoint.is_file() for r in runs))

    def test_checkpoint_precedence_and_root_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "model/ddpg/image/best_model.pt"
            shipped = root / "pretrained_model/ddpg/image/best_model.pt"
            explicit = root / "my-checkpoint.pt"
            for path in (legacy, shipped, explicit):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            with patch.object(run_experiment, "REPO_ROOT", root):
                args = run_experiment.ExperimentArgs(mpc_config=Path("config/custom.yaml"), solver_directory=Path("solver"))
                selected = run_experiment.resolve_runs(args)[0]
                self.assertEqual(selected.checkpoint, legacy)
                self.assertEqual(selected.mpc_config, root / "config/custom.yaml")
                self.assertEqual(selected.solver_directory, root / "solver")
                legacy.unlink()
                self.assertEqual(run_experiment.resolve_runs(args)[0].checkpoint, shipped)
                args.checkpoint = Path("my-checkpoint.pt")
                self.assertEqual(run_experiment.resolve_runs(args)[0].checkpoint, explicit)

    def test_evaluation_override_requires_one_selected_method(self) -> None:
        args = run_experiment.ExperimentArgs(workflow="ddpg-eval", observation="ray")
        with self.assertRaisesRegex(ValueError, "decision"):
            run_experiment.resolve_runs(args)
        args.decision = "hybrid"
        run = run_experiment.resolve_runs(args)[0]
        self.assertEqual((run.decision_mode, run.rl_index), (2, 1))

        default_mpc = run_experiment.resolve_runs(
            run_experiment.ExperimentArgs(workflow="ddpg-eval", decision="mpc")
        )[0]
        self.assertEqual((default_mpc.decision_mode, default_mpc.rl_index), (0, 1))

    def test_irrelevant_options_and_unsupported_case_are_rejected(self) -> None:
        cases = (
            run_experiment.ExperimentArgs(workflow="mpc-demo", observation="image"),
            run_experiment.ExperimentArgs(workflow="mpc-demo", case_index=0),
            run_experiment.ExperimentArgs(workflow="ddpg-demo", case_index=5),
            run_experiment.ExperimentArgs(workflow="dqn-demo", trials=2),
            run_experiment.ExperimentArgs(workflow="ddpg-demo", max_steps=0),
            run_experiment.ExperimentArgs(workflow="ddpg-eval", checkpoint=Path("model.pt")),
        )
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                run_experiment.resolve_runs(args)

    def test_default_solver_base_is_absolute_from_another_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config").mkdir()
            shutil.copyfile(ROOT / "config/mpc_default.yaml", root / "config/mpc_default.yaml")
            native = root / "mpc_solver/navi_default/navi_default.so"
            native.parent.mkdir(parents=True)
            native.touch()
            elsewhere = root / "elsewhere"
            elsewhere.mkdir()
            old_cwd = Path.cwd()
            try:
                os.chdir(elsewhere)
                with patch.object(run_experiment, "REPO_ROOT", root):
                    args = run_experiment.ExperimentArgs(workflow="mpc-demo")
                    output, solver_base = run_experiment._prepare(args, run_experiment.resolve_runs(args))
                self.assertIsNone(output)
                self.assertEqual(solver_base, root / "mpc_solver")
                self.assertEqual(Path.cwd(), elsewhere)
            finally:
                os.chdir(old_cwd)


class HelpAndBuildTests(unittest.TestCase):
    def test_help_needs_no_numerical_runtime(self) -> None:
        for script in ("run_experiment.py", "build_solver.py", "smoke.py"):
            with self.subTest(script=script):
                code = (
                    "import importlib.abc, importlib.util, sys; "
                    "blocked={'torch','numpy','gymnasium','casadi','opengen'}; "
                    "class_source='class Block(importlib.abc.MetaPathFinder):\\n'"
                    "' def find_spec(self, fullname, path=None, target=None):\\n'"
                    "'  if fullname.split(\".\")[0] in blocked: raise RuntimeError(fullname)\\n'; "
                    "exec(class_source); sys.meta_path.insert(0, Block()); "
                    f"spec=importlib.util.spec_from_file_location('tested_script', {str(ROOT / 'scripts' / script)!r}); "
                    "mod=importlib.util.module_from_spec(spec); sys.modules['tested_script']=mod; "
                    "spec.loader.exec_module(mod); mod.main(['--help'])"
                )
                process = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd="/tmp")
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertIn("--help", process.stdout)

    def test_build_refuses_existing_optimizer_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "candidate/navi_default"
            destination.mkdir(parents=True)
            (root / "config").mkdir()
            shutil.copyfile(ROOT / "config/mpc_default.yaml", root / "config/mpc_default.yaml")
            with patch.object(build_solver, "REPO_ROOT", root):
                args = build_solver.BuildArgs(build_directory=Path("candidate"))
                with self.assertRaisesRegex(FileExistsError, "navi_default"):
                    build_solver.build_solver(args)


class TrainingCliChecks(unittest.TestCase):
    def _run_lightweight(self, script: Path, *args: str) -> subprocess.CompletedProcess[str]:
        code = """
import builtins, runpy, sys
real_import = builtins.__import__
blocked = ('matplotlib', 'numpy', 'torch', 'gymnasium', 'stable_baselines3', 'drl_alg', 'drl_env')
def guarded_import(name, *args, **kwargs):
    if name.startswith(blocked):
        raise AssertionError('Heavy import during config-only CLI: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
sys.argv = [sys.argv[1], *sys.argv[2:]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
        return subprocess.run(
            [sys.executable, "-c", code, str(script), *args],
            cwd=tempfile.gettempdir(),
            env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
            text=True,
            capture_output=True,
        )

    def _load_training(self):
        spec = importlib.util.spec_from_file_location("training_for_checks", (ROOT / "src/training.py"))
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_help_and_config_save_skip_heavy_runtime(self):
        help_result = self._run_lightweight((ROOT / "src/training.py"), "--help")
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--evaluation-episodes", help_result.stdout)
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "input.yaml"
            saved = Path(temp) / "saved.yaml"
            config.write_text("mode: cluster\nindex: 1\nrun_vers: 3\nunknown: ignored\n")
            result = self._run_lightweight(
                (ROOT / "scripts/train.py"),
                "--config", str(config), "--run-vers", "5",
                "--save-config", str(saved),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            import yaml
            data = yaml.safe_load(saved.read_text())
            self.assertEqual((data["mode"], data["index"], data["run_vers"]), ("cluster", 1, 5))
            self.assertNotIn("unknown", data)
            self.assertNotIn("save_config", data)

    def test_parser_overrides_yaml_and_preserves_run_defaults(self):
        training = self._load_training()
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "input.yaml"
            config.write_text("mode: cluster\nindex: 1\nrun_vers: 3\nunknown: ignored\n")
            cfg = training._parse_args(["--config", str(config), "--run-vers", "5"])
        self.assertEqual((cfg.mode, cfg.index, cfg.run_vers), ("cluster", 1, 5))
        self.assertEqual(training.resolve_run_settings(cfg)[:2],
                         ("./Model/training/variant-1/run5", 7_000_000))
        self.assertEqual(training.resolve_run_settings(training.TrainingConfig())[:2],
                         ("./Model/training/variant-0/run0", 100_000))

    def test_bounded_nonrendered_evaluation_and_positive_limit(self):
        training = self._load_training()
        class Env:
            def __init__(self):
                self.resets = self.steps = self.renders = self.closes = 0
            def reset(self):
                self.resets += 1
                return 0, {}
            def step(self, action):
                self.steps += 1
                return 0, 1.0, True, False, {}
            def render(self):
                self.renders += 1
            def close(self):
                self.closes += 1
        class Model:
            @classmethod
            def load(cls, path, env):
                return cls()
            def predict(self, obs, deterministic):
                return 0, None
        env = Env()
        cfg = training.TrainingConfig(mode="cluster", evaluation=True,
                                      evaluation_episodes=2, render=False)
        with mock.patch.object(training, "make_env_eval", return_value=env), \
             mock.patch.object(training, "make_algorithm", return_value=Model), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            training.evaluate(cfg, training.VARIANTS[1], "unused")
        self.assertEqual((env.resets, env.steps, env.renders, env.closes), (2, 2, 0, 1))
        self.assertIn("mean=1.0", output.getvalue())
        with self.assertRaises(ValueError):
            training._parse_args(["--evaluation-episodes", "0"])


class RecordTests(unittest.TestCase):
    def test_numeric_streams_remain_separate_and_outputs_do_not_overwrite(self):
        from util.run_records import EpisodeRecord, write_episode

        first = EpisodeRecord()
        second = EpisodeRecord()
        first.environment_states.append(np.array([1., 2., 3., 4., 5.]))
        first.mpc_actions.append(np.array([.2, .1]))
        first.rl_actions.append(np.asarray(4))
        first.budget_exhausted = True
        first.termination = 'step_limit'
        self.assertEqual(second.environment_states, [])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'episode.npz'
            write_episode(first, path)
            with np.load(path, allow_pickle=False) as result:
                np.testing.assert_array_equal(result['environment_states'], [[1, 2, 3, 4, 5]])
                np.testing.assert_array_equal(result['mpc_actions'], [[.2, .1]])
                np.testing.assert_array_equal(result['rl_actions'], [4])
                self.assertEqual(result['mpc_states'].size, 0)
                self.assertTrue(bool(result['budget_exhausted']))
                self.assertFalse(bool(result['success']))
                self.assertEqual(str(result['termination']), 'step_limit')
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_episode(second, path)
            self.assertEqual(path.read_bytes(), before)


class ReviewFixTests(unittest.TestCase):

    def test_one_step_evaluation_keeps_record_without_fabricating_metrics(self):
        import evaluation
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            result = ([1.0], False, [(0.0, 0.0), (0.1, 0.0)], [(0.0, 0.0), (1.0, 0.0)], [(0.0, 0.0), (0.1, 0.0)], [[(2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0)]])
            args = run_experiment.ExperimentArgs(workflow='ddpg-eval', decision='hybrid', max_steps=1, trials=1, output_dir=output)
            with patch.object(run_experiment, '_prepare', return_value=(output, Path('/unused'))), patch.object(evaluation, 'main_process', return_value=result), contextlib.redirect_stdout(io.StringIO()) as printed:
                run_experiment.run_experiment(args)
            self.assertTrue((output / 'trial_001_hyb-ddpg-v.npz').is_file())
            self.assertIn('unavailable', printed.getvalue().lower())

    def test_cached_solver_conflict_rejected_before_output_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'config.yaml'
            config.write_text('optimizer_name: navi_default\nbuild_directory: solver\n')
            directory = root / 'solver/navi_default'
            directory.mkdir(parents=True)
            binding = directory / 'navi_default.so'
            binding.write_bytes(b'binding identity fixture')
            output = root / 'output'
            args = run_experiment.ExperimentArgs(workflow='mpc-demo', mpc_config=config, solver_directory=root / 'solver', output_dir=output)
            runs = run_experiment.resolve_runs(args)
            with patch.dict(sys.modules, {'navi_default': types.SimpleNamespace(__file__='/different/navi_default.so')}):
                with self.assertRaisesRegex(RuntimeError, 'process'):
                    run_experiment._prepare(args, runs)
            self.assertFalse(output.exists())

    def test_fleet_completion_classifies_each_robot(self):
        from _scenario_simulator import Simulator
        cfg = types.SimpleNamespace(Nstcobs=0, nstcobs=12, Ndynobs=0, ndynobs=6, action_steps=1)

        class Generator:

            def __init__(self, done):
                self.state = np.zeros(3)
                self.past_actions = []
                self.final_goal = np.zeros(3)
                self.done = done

            def load_init_state(self, *args):
                pass

            def set_work_mode(self, *args):
                pass

            def set_ref_trajectory(self, *args):
                pass

            def get_local_ref_traj(self):
                return (np.zeros((1, 3)), None)

            def run_step(self, *args, **kwargs):
                self.past_actions.append(np.zeros(2))
                return ([np.zeros(2)], np.zeros((1, 3)), 0.0)

            def check_termination_condition(self, *args):
                return self.done

        def robot(done):
            return types.SimpleNamespace(start=np.zeros(3), end=np.zeros(3), mode='work', ref_path=[], color='b', traj_gen=Generator(done), done=False, pred_states=None, budget_exhausted=False, termination='unknown')
        sim = Simulator.__new__(Simulator)
        sim.config = cfg
        sim.N_hor = 1
        sim.ts = 0.2
        sim.use_tcp = False
        sim.robot_dict = {0: robot(True), 1: robot(False)}
        sim.get_other_robot_states = lambda r: []
        scanner = types.SimpleNamespace(get_full_obstacle_list=lambda **kwargs: [])
        result = sim.run(lambda: ([], []), scanner, max_steps=1)
        self.assertTrue(result[0].done)
        self.assertEqual(result[0].termination, 'completed')
        self.assertFalse(result[0].budget_exhausted)
        self.assertEqual(result[1].termination, 'step_limit')
        self.assertTrue(result[1].budget_exhausted)

if __name__ == "__main__":
    unittest.main()
