"""Build the existing MPC optimizer into a new candidate directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class BuildArgs:
    config_file: Path = Path("config/mpc_default.yaml")
    build_directory: Path = Path("mpc_solver/candidate")
    use_tcp: bool = False


def _root_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def build_solver(args: BuildArgs) -> None:
    config_path = _root_path(args.config_file)
    destination = _root_path(args.build_directory)
    if not config_path.is_file():
        raise FileNotFoundError(f"MPC config does not exist: {config_path}")

    import yaml

    config_values = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config_values, dict):
        raise TypeError("MPC YAML root must be a mapping")
    optimizer_name = config_values.get("optimizer_name", "navi_default")
    if not isinstance(optimizer_name, str):
        raise TypeError("MPC optimizer_name must be a string")
    optimizer_directory = destination / optimizer_name
    if optimizer_directory.exists():
        raise FileExistsError(f"Refusing to overwrite existing optimizer: {optimizer_directory}")

    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from mpc_traj_tracker.config import MPCConfig

    config = MPCConfig.from_yaml(str(config_path))
    destination.mkdir(parents=True, exist_ok=True)
    config.build_directory = str(destination)

    from motion_model.motion_model import unicycle_model
    from mpc_traj_tracker.solver_generator import MpcModule

    print(f"Building {config.optimizer_name} in {destination}")
    MpcModule(config).build(unicycle_model, use_tcp=args.use_tcp)


def main(argv: Sequence[str] | None = None) -> None:
    import tyro

    build_solver(tyro.cli(BuildArgs, args=argv))


if __name__ == "__main__":
    main()
