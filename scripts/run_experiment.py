"""Typed entry point for the repository's existing experiment loops."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import importlib
from importlib.machinery import EXTENSION_SUFFIXES, PathFinder
from importlib import metadata as package_metadata
from pathlib import Path
import subprocess
import sys
from typing import TYPE_CHECKING, Literal, Sequence

if TYPE_CHECKING:
    from util.run_records import EpisodeRecord


REPO_ROOT = Path(__file__).resolve().parents[1]
Workflow = Literal["mpc-demo", "ddpg-demo", "dqn-demo", "ddpg-eval"]
Decision = Literal["mpc", "rl", "hybrid"]
Observation = Literal["image", "ray"]
Visualization = Literal["default", "on", "off"]


@dataclass
class ExperimentArgs:
    """Select a retained demo or quantitative evaluation workflow."""

    workflow: Workflow = "ddpg-demo"
    decision: Decision | None = None
    observation: Observation | None = None
    scene: tuple[int, int, int] | None = None
    case_index: int | None = None
    visualization: Visualization = "default"
    max_steps: int | None = None
    trials: int | None = None
    checkpoint: Path | None = None
    mpc_config: Path = Path("config/mpc_default.yaml")
    solver_directory: Path | None = None
    seed: int | None = None
    output_dir: Path | None = None


@dataclass(frozen=True)
class RunSpec:
    workflow: Workflow
    decision_mode: Literal[0, 1, 2]
    rl_index: Literal[0, 1] | None
    scene_option: tuple[int, int, int] | None
    case_index: int | None
    to_plot: bool
    max_steps: int | None
    checkpoint: Path | None
    mpc_config: Path
    solver_directory: Path | None


def parse_args(argv: Sequence[str] | None = None) -> ExperimentArgs:
    import tyro

    return tyro.cli(ExperimentArgs, args=argv)


def _root_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _checkpoint(algorithm: str, observation: Observation, explicit: Path | None) -> Path:
    if explicit is not None:
        selected = _root_path(explicit)
        if not selected.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {selected}")
        return selected
    for base in ("model", "pretrained_model"):
        candidate = REPO_ROOT / base / algorithm / observation / "best_model.pt"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No {algorithm}/{observation} best_model.pt in model/ or pretrained_model/")


def resolve_runs(args: ExperimentArgs) -> list[RunSpec]:
    """Resolve entry presets and reject options that the chosen loop cannot use."""
    if args.workflow not in ("mpc-demo", "ddpg-demo", "dqn-demo", "ddpg-eval"):
        raise ValueError(f"Unknown workflow: {args.workflow}")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if args.trials is not None and args.trials <= 0:
        raise ValueError("trials must be positive")
    if args.visualization not in ("default", "on", "off"):
        raise ValueError("visualization must be default, on, or off")

    config = _root_path(args.mpc_config)
    solver = _root_path(args.solver_directory) if args.solver_directory is not None else None
    to_plot = args.visualization == "on" or (args.visualization == "default" and args.workflow != "ddpg-eval")

    if args.workflow == "mpc-demo":
        if any(value is not None for value in (args.decision, args.observation, args.scene, args.checkpoint)):
            raise ValueError("MPC demo does not accept decision, observation, scene, or checkpoint")
        if args.trials not in (None, 1):
            raise ValueError("MPC demo runs once; trials applies to evaluation")
        case = 4 if args.case_index is None else args.case_index
        if case == 0:
            raise ValueError("MPC case 0 is unsupported")
        return [RunSpec(args.workflow, 0, None, None, case, to_plot, args.max_steps, None, config, solver)]

    if args.case_index is not None:
        raise ValueError("case_index applies only to the MPC demo")
    if args.workflow != "ddpg-eval" and args.trials not in (None, 1):
        raise ValueError("trials applies only to evaluation")

    scene = args.scene or ((1, 1, 2) if args.workflow == "dqn-demo" else (1, 3, 2))
    algorithm = "dqn" if args.workflow == "dqn-demo" else "ddpg"
    if args.workflow == "ddpg-eval" and args.decision is None:
        if args.observation is not None or args.checkpoint is not None:
            raise ValueError("Select an evaluation decision before overriding observation or checkpoint")
        selections: list[tuple[Decision, Observation]] = [("mpc", "ray"), ("rl", "image"), ("hybrid", "image")]
    else:
        decision = args.decision or "hybrid"
        default_observation: Observation = "ray" if args.workflow == "ddpg-eval" and decision == "mpc" else "image"
        selections = [(decision, args.observation or default_observation)]

    runs: list[RunSpec] = []
    for decision, observation in selections:
        if decision not in ("mpc", "rl", "hybrid") or observation not in ("image", "ray"):
            raise ValueError("Unknown decision or observation")
        mode: Literal[0, 1, 2] = {"mpc": 0, "rl": 1, "hybrid": 2}[decision]
        rl_index: Literal[0, 1] = 0 if observation == "image" else 1
        checkpoint = _checkpoint(algorithm, observation, args.checkpoint)
        runs.append(RunSpec(args.workflow, mode, rl_index, scene, None, to_plot, args.max_steps, checkpoint, config, solver))
    return runs


def _method_label(spec: RunSpec) -> str:
    if spec.workflow == "mpc-demo" or spec.decision_mode == 0:
        return "MPC"
    prefix = "HYB-" if spec.decision_mode == 2 else ""
    algorithm = "DQN" if spec.workflow == "dqn-demo" else "DDPG"
    suffix = "V" if spec.rl_index == 0 else "L"
    return f"{prefix}{algorithm}-{suffix}"


def _add_import_paths() -> None:
    for directory in (REPO_ROOT / "test", REPO_ROOT / "tests/demos", REPO_ROOT / "src"):
        string = str(directory)
        if directory.is_dir() and string not in sys.path:
            sys.path.insert(0, string)


def _native_artifact(spec: RunSpec, optimizer_name: str, build_directory: Path) -> Path:
    solver_base = spec.solver_directory or _root_path(build_directory)
    optimizer_dir = solver_base / optimizer_name
    if not optimizer_dir.is_dir():
        raise FileNotFoundError(f"MPC optimizer directory does not exist: {optimizer_dir}")
    for suffix in EXTENSION_SUFFIXES:
        candidate = optimizer_dir / f"{optimizer_name}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Compatible MPC native binding is missing under: {optimizer_dir}")


def _verify_native_import(optimizer_name: str, native: Path) -> None:
    loaded = sys.modules.get(optimizer_name)
    if loaded is not None:
        origin = getattr(loaded, "__file__", None)
    else:
        found = PathFinder.find_spec(optimizer_name, [*sys.path, str(native.parent)])
        origin = found.origin if found is not None else None
    if origin is None or Path(origin).resolve() != native.resolve():
        raise RuntimeError(f"MPC binding import would not use selected native artifact: {native}; found {origin}. Select this solver in a fresh process")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> tuple[str, bool]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return (commit.stdout.strip() if commit.returncode == 0 else "unknown", bool(status.stdout.strip()) if status.returncode == 0 else False)


def _versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for package in ("numpy", "torch", "gymnasium", "stable-baselines3", "casadi", "opengen", "tyro"):
        try:
            versions[package] = package_metadata.version(package)
        except package_metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def _prepare(args: ExperimentArgs, runs: list[RunSpec]) -> tuple[Path | None, Path]:
    """Validate artifacts and reserve optional output before any simulation starts."""
    import yaml

    _add_import_paths()

    config_path = runs[0].mpc_config
    if not config_path.is_file():
        raise FileNotFoundError(f"MPC config does not exist: {config_path}")
    mpc_yaml = config_path.read_text(encoding="utf-8")
    config_values = yaml.safe_load(mpc_yaml) or {}
    if not isinstance(config_values, dict):
        raise TypeError("MPC YAML root must be a mapping")
    build_directory = Path(config_values.get("build_directory", "mpc_solver"))
    optimizer_name = config_values.get("optimizer_name", "navi_default")
    if not isinstance(optimizer_name, str):
        raise TypeError("MPC optimizer_name must be a string")
    native = _native_artifact(runs[0], optimizer_name, build_directory)
    _verify_native_import(optimizer_name, native)
    solver_base = runs[0].solver_directory or _root_path(build_directory)
    artifacts: dict[str, Path] = {"mpc_config": config_path, "mpc_native": native}
    for index, run in enumerate(runs):
        if run.checkpoint is not None:
            artifacts[f"checkpoint_{_method_label(run)}"] = run.checkpoint
    hash_cache: dict[Path, str] = {}
    hashes: dict[str, str] = {}
    for key, path in artifacts.items():
        if path not in hash_cache:
            hash_cache[path] = _sha256(path)
        hashes[key] = hash_cache[path]

    output = _root_path(args.output_dir) if args.output_dir is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        _add_import_paths()
        from util.run_records import RunMetadata, write_metadata

        commit, dirty = _git_state()
        metadata = RunMetadata(
            workflow=args.workflow,
            methods=[_method_label(run) for run in runs],
            scene=runs[0].scene_option,
            case_index=runs[0].case_index,
            seed=args.seed,
            trials=args.trials or (5 if args.workflow == "ddpg-eval" else 1),
            max_steps=args.max_steps,
            visualization=runs[0].to_plot,
            commit=commit,
            dirty=dirty,
            versions=_versions(),
            artifact_paths={key: str(path) for key, path in artifacts.items()},
            artifact_sha256=hashes,
            mpc_yaml=mpc_yaml,
            details={"solver_directory": str(solver_base)},
        )
        write_metadata(metadata, output / "metadata.json")
    return output, solver_base


def _seed_explicitly(seed: int | None) -> None:
    if seed is None:
        return
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _write_fleet(playback: dict, output: Path) -> None:
    import numpy as np

    arrays: dict[str, np.ndarray] = {}
    for robot_id, robot in playback.items():
        prefix = f"robot_{robot_id}"
        generator = robot.traj_gen
        states = [*generator.past_states, generator.state]
        arrays[f"{prefix}_states"] = np.asarray(states)
        arrays[f"{prefix}_actions"] = np.asarray(generator.past_actions)
        arrays[f"{prefix}_costs"] = np.asarray(generator.cost_timelist)
        arrays[f"{prefix}_termination"] = np.asarray(getattr(robot, "termination", "unknown"))
        arrays[f"{prefix}_budget_exhausted"] = np.asarray(getattr(robot, "budget_exhausted", False))
        arrays[f"{prefix}_done"] = np.asarray(robot.done)
    with (output / "trial_001_mpc_fleet.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _check_episode(record: EpisodeRecord, timing: Sequence[float], label: str) -> None:
    if record.termination == "solver_error":
        raise RuntimeError(f"{label}: solver_error; see the episode record and solver output")
    if not timing:
        raise RuntimeError(f"{label}: empty control trace")


def _check_fleet(playback: dict) -> None:
    if not playback:
        raise RuntimeError("MPC fleet: empty robot trace")
    for robot_id, robot in playback.items():
        if not robot.traj_gen.past_actions:
            raise RuntimeError(f"MPC fleet robot {robot_id}: empty control trace")


def run_experiment(args: ExperimentArgs) -> None:
    runs = resolve_runs(args)
    output, solver_base = _prepare(args, runs)
    runs = [replace(spec, solver_directory=solver_base) for spec in runs]
    _seed_explicitly(args.seed)
    _add_import_paths()
    from util.run_records import EpisodeRecord, write_episode

    if args.workflow == "mpc-demo":
        from test_mpc import Args as MpcArgs, run_simulation

        spec = runs[0]
        print(f"MPC config: {spec.mpc_config}")
        playback = run_simulation(MpcArgs(config_file=spec.mpc_config.name, build=False, plot=spec.to_plot,
                                          case_index=spec.case_index, mpc_config=spec.mpc_config,
                                          solver_directory=spec.solver_directory, max_steps=spec.max_steps))
        if output is not None:
            _write_fleet(playback, output)
        _check_fleet(playback)
        return

    if args.workflow == "ddpg-eval":
        from evaluation import main_process
        from helper import Metrics

        metrics = {_method_label(spec): Metrics(mode=_method_label(spec)) for spec in runs}
        trials = args.trials or 5
        unavailable_metrics: set[str] = set()
        for trial in range(trials):
            print(f"Trial {trial + 1}/{trials}")
            for spec in runs:
                label = _method_label(spec)
                print(f"{label} checkpoint: {spec.checkpoint}")
                record = EpisodeRecord()
                timing, success, actions, reference, actual, obstacles = main_process(
                    rl_index=spec.rl_index, decision_mode=spec.decision_mode, to_plot=spec.to_plot,
                    scene_option=spec.scene_option, checkpoint=spec.checkpoint, mpc_config=spec.mpc_config,
                    solver_directory=spec.solver_directory, max_steps=spec.max_steps, record=record,
                )
                if output is not None:
                    write_episode(record, output / f"trial_{trial + 1:03d}_{label.lower()}.npz")
                _check_episode(record, timing, label)
                # A short probe can lack samples required by the retained
                # variance and second-difference formulas. Keep its raw record;
                # do not fabricate aggregate values or average a subset of trials.
                if len(timing) < 2 or len(actions) < 3:
                    unavailable_metrics.add(label)
                else:
                    metrics[label].add_trial_result(timing, success, actions, reference, actual, obstacles)
        for label, metric in metrics.items():
            print(label)
            if label in unavailable_metrics:
                print("Metrics unavailable: at least one trial has insufficient samples.")
            else:
                print(metric.get_average(3))
        return

    module = importlib.import_module("test_ddpg" if args.workflow == "ddpg-demo" else "test_dqn")
    spec = runs[0]
    print(f"{_method_label(spec)} checkpoint: {spec.checkpoint}")
    record = EpisodeRecord()
    timing = module.main(rl_index=spec.rl_index, decision_mode=spec.decision_mode, to_plot=spec.to_plot,
                         scene_option=spec.scene_option, checkpoint=spec.checkpoint, mpc_config=spec.mpc_config,
                         solver_directory=spec.solver_directory, max_steps=spec.max_steps, record=record)
    if output is not None:
        write_episode(record, output / f"trial_001_{_method_label(spec).lower()}.npz")
    _check_episode(record, timing, _method_label(spec))


def main(argv: Sequence[str] | None = None) -> None:
    run_experiment(parse_args(argv))


if __name__ == "__main__":
    main()
