"""Bounded real solver/policy checks. This is not a training benchmark."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class SmokeArgs:
    output_dir: Path
    checkpoint_root: Path = Path("pretrained_model")
    solver_directory: Path = Path("mpc_solver/candidate")
    steps: int = 10
    seed: int = 0
    learning_probe: bool = False


def _root_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _verify_smoke_run(run_dir: Path) -> None:
    """Require real control samples and a known outcome from each bounded run."""
    import numpy as np

    records = sorted(run_dir.glob("trial_*.npz"))
    if not records:
        raise RuntimeError(f"Smoke run has no episode records: {run_dir}")
    for path in records:
        with np.load(path, allow_pickle=False) as data:
            if path.name.endswith("_mpc_fleet.npz"):
                action_keys = [key for key in data.files if key.endswith("_actions")]
                if not action_keys:
                    raise RuntimeError(f"Smoke fleet has no robot traces: {path}")
                for key in action_keys:
                    prefix = key.removesuffix("_actions")
                    if data[key].size == 0:
                        raise RuntimeError(f"Smoke fleet has an empty control trace: {path} {prefix}")
                    status = str(data[f"{prefix}_termination"].item())
                    if status not in ("completed", "step_limit"):
                        raise RuntimeError(f"Smoke fleet has invalid termination {status}: {path} {prefix}")
            else:
                status = str(data["termination"].item())
                if status not in ("success", "collision", "step_limit", "terminated"):
                    raise RuntimeError(f"Smoke episode has invalid termination {status}: {path}")
                if "timing_ms" not in data.files or data["timing_ms"].size == 0:
                    raise RuntimeError(f"Smoke episode has an empty control trace: {path}")


def run_learning_probe(output_dir: Path, seed: int = 0) -> dict[str, object]:
    """Eight disposable ray/CPU steps with probe-only optimization settings."""
    import gymnasium as gym
    import numpy as np
    import torch
    import drl_env
    from drl_alg.map import generate_map_eval
    from drl_alg.per_ddpg import PerDDPG

    output_dir.mkdir(parents=True, exist_ok=False)
    env = gym.make("TrajectoryPlannerEnvironmentRaysReward-v0", generate_map=generate_map_eval)
    try:
        model = PerDDPG("MultiInputPolicy", env, policy_kwargs={"net_arch": [16, 16]},
                        buffer_size=128, batch_size=2, learning_starts=0,
                        train_freq=1, gradient_steps=1, device="cpu", seed=seed)
        before = {name: value.detach().clone() for name, value in model.policy.named_parameters()}
        model.learn(total_timesteps=8)
        changed = any(not torch.equal(before[name], value) for name, value in model.policy.named_parameters())
        if model._n_updates < 1 or not changed:
            raise AssertionError("The disposable probe did not update policy parameters")
        model.save(str(output_dir / "best_model"))
        # This is an actual SB3 archive produced by the probe, not a converted
        # shipped policy. Exercise the retained evaluation path without drawing.
        from training import TrainingConfig, evaluate, VARIANTS
        evaluate(TrainingConfig(index=1, evaluation=True, evaluation_episodes=1, render=False),
                 VARIANTS[1], str(output_dir))
        result: dict[str, object] = {"steps": int(model.num_timesteps), "updates": int(model._n_updates),
                                    "parameters_changed": changed, "archive_evaluation": "passed",
                                    "scope": "probe only; no convergence or historical archive claim"}
        (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        env.close()


def run_smoke(output_dir: Path, checkpoint_root: Path, solver_directory: Path,
              steps: int = 10, seed: int = 0) -> None:
    """Exercise shipped policies, hybrid demos, evaluation, and two-robot MPC."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    from run_experiment import ExperimentArgs, run_experiment, _add_import_paths, _seed_explicitly
    _add_import_paths()
    import importlib
    import numpy as np

    output_dir = _root_path(output_dir)
    checkpoint_root = _root_path(checkpoint_root)
    solver_directory = _root_path(solver_directory)
    output_dir.mkdir(parents=True, exist_ok=False)
    prediction_arrays: dict[str, np.ndarray] = {}
    checkpoint_hashes: dict[str, str] = {}
    for algorithm in ("ddpg", "dqn"):
        module = importlib.import_module("test_" + algorithm)
        for index, modality in enumerate(("image", "ray")):
            _seed_explicitly(seed)
            checkpoint = checkpoint_root / algorithm / modality / "best_model.pt"
            model, env = module.load_rl_model_env(module.generate_map(1, 1, 2), index, checkpoint=checkpoint)
            try:
                observation, _ = env.reset(seed=seed)
                action, _ = model.predict(observation, deterministic=True)
                key = f"{algorithm}_{modality}"
                prediction_arrays[key + "_action"] = np.asarray(action).copy()
                for name, value in observation.items():
                    prediction_arrays[key + "_" + name] = np.asarray(value).copy()
                checkpoint_hashes[key] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            finally:
                env.close()
    np.savez_compressed(output_dir / "fixed_policy_inputs.npz", **prediction_arrays)
    (output_dir / "checkpoints.json").write_text(json.dumps(checkpoint_hashes, indent=2) + "\n")

    for workflow, algorithm in (("ddpg-demo", "ddpg"), ("dqn-demo", "dqn"), ("ddpg-eval", "ddpg")):
        run_dir = output_dir / workflow
        run_experiment(ExperimentArgs(workflow=workflow, decision="hybrid", observation="image",
                                      checkpoint=checkpoint_root / algorithm / "image/best_model.pt",
                                      solver_directory=solver_directory, max_steps=steps, trials=1,
                                      visualization="off", seed=seed, output_dir=run_dir))
        _verify_smoke_run(run_dir)
    fleet_dir = output_dir / "fleet"
    run_experiment(ExperimentArgs(workflow="mpc-demo", case_index=5, solver_directory=solver_directory,
                                  max_steps=steps, visualization="off", seed=seed,
                                  output_dir=fleet_dir))
    _verify_smoke_run(fleet_dir)
    print(f"Real-workflow smoke evidence: {output_dir}")


def main(argv: Sequence[str] | None = None) -> None:
    import tyro
    args = tyro.cli(SmokeArgs, args=argv)
    run_smoke(args.output_dir, args.checkpoint_root, args.solver_directory, args.steps, args.seed)
    if args.learning_probe:
        run_learning_probe(_root_path(args.output_dir) / "learning-probe", args.seed)


if __name__ == "__main__":
    main()
