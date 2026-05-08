"""Typed configuration dataclasses and enums for drl_mpc_nav."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class DecisionMode(StrEnum):
    """Selects which planner produces the final control command."""

    MPC = "mpc"
    DDPG = "ddpg"
    HYBRID = "hybrid"


class RLSensor(StrEnum):
    """Sensor modality used by the RL policy."""

    IMAGE = "image"
    RAY = "ray"


@dataclass
class EvalConfig:
    """Configuration for a single evaluation run.

    Attributes:
        rl_index:       0 = image-based DDPG, 1 = ray-based DDPG.
        decision_mode:  Which planner to use.
        max_steps:      Maximum number of environment steps per episode.
        dyn_obs_size:   Diameter (m) assumed for dynamic-obstacle bounding boxes.
        new_rl_ref:     Whether to roll out the RL policy for horizon planning
                        (True) or use single-step decay (False).
        scene:          (scene, sub_scene, sub_scene_option) tuple.
        model_dir:      Directory containing ``best_model.pt`` files.
        mpc_config:     Path to the MPC YAML configuration file.
        plot:           Whether to render the environment during evaluation.
        seed:           Random seed for reproducibility.
        verbose:        Print per-step timing and mode information.
    """

    rl_index: int = 1
    decision_mode: DecisionMode = DecisionMode.HYBRID
    max_steps: int = 200
    dyn_obs_size: float = 0.8 + 0.8
    new_rl_ref: bool = True
    scene: tuple[int, int, int] = (1, 3, 2)
    model_dir: Path = Path("model/ddpg")
    mpc_config: Path = Path("configs/mpc_default.yaml")
    plot: bool = False
    seed: int = 0
    verbose: bool = False


@dataclass
class TrainConfig:
    """Configuration for a training run.

    Attributes:
        variant_index:    Which pre-defined variant to use (0–5).
        run_version:      Integer run identifier appended to the save path.
        total_timesteps:  Total number of environment interactions.
        n_envs:           Number of parallel training environments.
        load_checkpoint:  Resume training from an existing checkpoint.
        output_dir:       Root directory for model checkpoints and logs.
        seed:             Random seed for reproducibility.
        device:           PyTorch device string (``"cpu"``, ``"cuda"``, ``"auto"``).
    """

    variant_index: int = 6
    run_version: int = 1
    total_timesteps: int = 100_000
    n_envs: int = 4
    load_checkpoint: bool = False
    output_dir: Path = Path("outputs/training")
    seed: int = 0
    device: str = "auto"


@dataclass
class MPCSolverBuildConfig:
    """Configuration for rebuilding the MPC solver.

    Attributes:
        config_file: Name of the YAML config file inside ``configs/``.
        use_tcp:     Whether to use the TCP interface to the solver build service.
        verbose:     Print solver build output.
    """

    config_file: str = "mpc_default.yaml"
    use_tcp: bool = False
    verbose: bool = True
