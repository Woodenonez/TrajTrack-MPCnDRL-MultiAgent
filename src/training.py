"""Unified training script for local and cluster modes.

Usage examples
--------------
# Use CLI args directly:
    python scripts/train.py --mode local --index 0 --run-vers 13

# Load defaults from YAML, then optionally override via CLI:
    python scripts/train.py --config config/training.yaml
    python scripts/train.py --config config/training.yaml --mode cluster --index 0

# Save your current settings back to YAML for later:
    python scripts/train.py --mode cluster --index 0 --run-vers 5 --save-config config/training.yaml
"""

from __future__ import annotations

import dataclasses
import sys
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, Optional, Sequence

import tyro
import yaml # type: ignore[import]

if TYPE_CHECKING:
    import gymnasium as gym
    from stable_baselines3 import DDPG
    from drl_env import MapDescription


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TrainingConfig:
    """Training configuration for the DRL-MPC agent."""

    mode: Literal["local", "cluster"] = "local"
    """Training mode: 'local' for interactive development, 'cluster' for HPC runs."""

    index: int = 0
    """Index into the variant table that selects the algorithm/environment combination."""

    run_vers: int = 0
    """Run version number, used to organise checkpoints under the model directory."""

    path: Optional[str] = None
    """Explicit output path.  When None the path is derived as
    './Model/training/variant-<index>/run<run_vers>'."""

    evaluation: bool = False
    """When True, load the best saved model and run evaluation instead of training."""

    evaluation_episodes: int | None = None
    """Optional positive episode limit for checkpoint evaluation."""

    render: bool = True
    """Display plots and frames during checkpoint evaluation."""

    tot_timesteps: Optional[int] = None
    """Total training timesteps.  Defaults to 1e5 for local, 7e6 for cluster."""

    n_cpu: int = 20
    """Number of parallel environments."""

    save_config: Optional[str] = None
    """If set, write the resolved config to this YAML path and exit."""

    @classmethod
    def from_yaml(cls, yaml_path: str) -> TrainingConfig:
        with open(yaml_path) as fh:
            data = yaml.safe_load(fh) or {}
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)
    
    def to_yaml(self, yaml_path: str) -> None:
        data = dataclasses.asdict(self)
        data.pop("save_config", None)  # don't persist the meta-option
        with open(yaml_path, "w") as fh:
            yaml.dump(data, fh, default_flow_style=False, sort_keys=False)
        print(f"Config saved to {yaml_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def plot_training_results(path: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        f = np.load(f"{path}/evaluations.npz")
    except Exception:
        f = np.load(path)

    mean = np.mean(f["results"], 1)
    max_ind = np.argmax(mean)

    plt.figure()
    plt.plot(f["timesteps"], mean)
    plt.plot(f["timesteps"][max_ind], mean[max_ind], "r*")
    plt.xlabel("Total number of steps taken")
    plt.ylabel("Mean return over %d evaluation episode" % len(f["results"][0]))
    plt.title("Training results")
    plt.show()

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Return a linear learning-rate schedule decaying from *initial_value*."""

    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func

def _make_generate_map(mode: Literal["local", "cluster"]) -> Callable[[], MapDescription]:
    """Return a generate_map callable appropriate for *mode*."""
    import drl_alg.map as map_api

    return map_api.choose_generator_for_mode(mode)


# ---------------------------------------------------------------------------
# Variant table (shared between modes)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Variant:
    algorithm: str
    env_name: str
    net_arch: list[int]
    per: bool
    device: str

VARIANTS: list[Variant] = [
    Variant(
        algorithm="DDPG",
        env_name="TrajectoryPlannerEnvironmentImgsReward-v0",
        net_arch=[64, 64],
        per=True,
        device="cuda",
    ),
    Variant(
        algorithm="DDPG",
        env_name="TrajectoryPlannerEnvironmentRaysReward-v0",
        net_arch=[16, 16],
        per=True,
        device="cpu",
    ),
]


# ---------------------------------------------------------------------------
# Main training / evaluation routine
# ---------------------------------------------------------------------------

def resolve_run_settings(cfg: TrainingConfig) -> tuple[str, int, Variant]:
    effective_path: str = cfg.path or f"./Model/training/variant-{cfg.index}/run{cfg.run_vers}"
    mode_default_timesteps = 100_000 if cfg.mode == "local" else 7_000_000
    tot_timesteps = cfg.tot_timesteps if cfg.tot_timesteps is not None else mode_default_timesteps
    variant = VARIANTS[cfg.index]
    return effective_path, int(tot_timesteps), variant


def make_env_kwargs(cfg: TrainingConfig) -> tuple[dict[str, Any], dict[str, Any], int]:
    import drl_alg.map as map_api

    generate_map = _make_generate_map(cfg.mode)

    if cfg.mode == "local":
        env_kwargs_train = {"generate_map": generate_map}
        env_kwargs_eval = {"generate_map": map_api.generate_map_eval}
        eval_freq_divisor = 100
    else:
        env_kwargs_train = {"generate_map": map_api.generate_simple_map_static}
        env_kwargs_eval = {"generate_map": map_api.generate_simple_map_static}
        eval_freq_divisor = 1000

    return env_kwargs_train, env_kwargs_eval, eval_freq_divisor


def make_env_eval(cfg: TrainingConfig, variant: Variant) -> gym.Env:
    import gymnasium as gym
    import drl_alg.map as map_api

    if cfg.mode == "local":
        return gym.make(variant.env_name, generate_map=map_api.generate_map_eval)
    return gym.make(variant.env_name, generate_map=map_api.generate_map_mpc(11))


def make_algorithm(variant: Variant) -> type[DDPG]:
    from stable_baselines3 import DDPG
    from drl_alg.per_ddpg import PerDDPG

    return PerDDPG if (variant.algorithm == "DDPG" and variant.per) else DDPG


def train(cfg: TrainingConfig, variant: Variant, effective_path: str, tot_timesteps: int) -> None:
    import numpy as np
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
    from stable_baselines3.common.vec_env import SubprocVecEnv

    env_kwargs_train, env_kwargs_eval, eval_freq_divisor = make_env_kwargs(cfg)

    vec_env = make_vec_env(
        variant.env_name,
        n_envs=cfg.n_cpu,
        seed=0,
        vec_env_cls=SubprocVecEnv if cfg.n_cpu > 1 else None,
        env_kwargs=env_kwargs_train,
    )
    vec_env_eval = make_vec_env(
        variant.env_name,
        n_envs=cfg.n_cpu,
        seed=0,
        vec_env_cls=SubprocVecEnv if cfg.n_cpu > 1 else None,
        env_kwargs=env_kwargs_eval,
    )

    assert vec_env.action_space.shape is not None
    n_actions = vec_env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
    )

    algorithm = make_algorithm(variant)
    eval_callback = EvalCallback(
        vec_env_eval,
        best_model_save_path=effective_path,
        log_path=effective_path,
        eval_freq=int(max((tot_timesteps / eval_freq_divisor) // cfg.n_cpu, 1)),
        n_eval_episodes=cfg.n_cpu,
    )
    
    model = algorithm(
        "MultiInputPolicy",
        vec_env,
        learning_rate=0.0001,
        buffer_size=int(1e6),
        learning_starts=100_000,
        gamma=0.98,
        gradient_steps=-1,
        action_noise=action_noise,
        policy_kwargs={"net_arch": variant.net_arch},
        verbose=1,
        device=variant.device,
    )
    
    if variant.device == "cuda" and model.device.type != "cuda":
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        if cfg.mode == "local":
            input("Press Enter to continue...")
        else:
            raise RuntimeError("CUDA requested but not available.")

    try:
        model.learn(
            total_timesteps=int(tot_timesteps),
            log_interval=4,
            progress_bar=True,
            callback=eval_callback,
        )
        model.save(f"{effective_path}/final_model")
    finally:
        vec_env.close()
        vec_env_eval.close()


def evaluate(cfg: TrainingConfig, variant: Variant, effective_path: str) -> None:
    from itertools import count
    from torch import no_grad

    if cfg.evaluation_episodes is not None and cfg.evaluation_episodes <= 0:
        raise ValueError("evaluation_episodes must be positive")

    env_eval = make_env_eval(cfg, variant)

    try:
        algorithm = make_algorithm(variant)
        model = algorithm.load(f"{effective_path}/best_model", env=env_eval)
        with no_grad():
            if cfg.mode == "local":
                if cfg.render:
                    plot_training_results(effective_path)
                episodes: Iterable[int] = (
                    count() if cfg.evaluation_episodes is None else range(cfg.evaluation_episodes)
                )
                for _ in episodes:
                    obs, *_ = env_eval.reset()
                    for i in range(1000):
                        action, _states = model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = env_eval.step(action)
                        if cfg.render and i % 3 == 0:
                            env_eval.render()
                        if terminated or truncated:
                            break
            else:
                rew_list = []
                for _ in range(cfg.evaluation_episodes or 1):
                    obs, *_ = env_eval.reset()
                    cum_ret = 0.0
                    for i in range(1000):
                        action, _states = model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = env_eval.step(action)
                        cum_ret += float(reward)
                        if cfg.render and i % 3 == 0:
                            env_eval.render()
                        if terminated or truncated:
                            print(cum_ret)
                            rew_list.append(cum_ret)
                            break
                print(f"mean={sum(rew_list) / len(rew_list)}")
    finally:
        env_eval.close()


def run(cfg: TrainingConfig) -> None:
    effective_path, tot_timesteps, variant = resolve_run_settings(cfg)

    if cfg.mode == "local":
        import gymnasium as gym
        from stable_baselines3.common.env_checker import check_env

        env_kwargs_train, *_ = make_env_kwargs(cfg)
        with gym.make(variant.env_name, **env_kwargs_train) as env:
            check_env(env)

    if cfg.evaluation:
        evaluate(cfg, variant, effective_path)
    else:
        train(cfg, variant, effective_path, tot_timesteps)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: Sequence[str] | None = None) -> TrainingConfig:
    """Parse a leading --config <yaml> (if present), then run tyro on the rest."""
    args: list[str] = list(sys.argv[1:] if argv is None else argv)

    # Extract --config <path> before tyro sees the argument list
    yaml_path: Optional[str] = None
    filtered_argv: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--config" and i + 1 < len(args):
            yaml_path = args[i + 1]
            i += 2
        else:
            filtered_argv.append(args[i])
            i += 1

    # Build defaults: start from dataclass defaults, then overlay YAML values
    defaults = TrainingConfig()
    if yaml_path is not None:
        defaults = TrainingConfig.from_yaml(yaml_path)

    # Let tyro parse the remaining argv, using the (possibly YAML-seeded) defaults
    cfg: TrainingConfig = tyro.cli(
        TrainingConfig,
        default=defaults,
        args=filtered_argv,
    )
    if cfg.evaluation_episodes is not None and cfg.evaluation_episodes <= 0:
        raise ValueError("evaluation_episodes must be positive")
    return cfg


def main(argv: Sequence[str] | None = None) -> None:
    cfg = _parse_args(argv)

    if cfg.save_config is not None:
        cfg.to_yaml(cfg.save_config)
        return

    # Print the resolved config for checking and logging
    print("Running configuration:")
    for field in dataclasses.fields(cfg):
        value = getattr(cfg, field.name)
        print(f" -- {field.name}: {value}")

    run(cfg)


if __name__ == "__main__":
    main()
