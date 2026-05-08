"""CLI entry point for training a DDPG agent.

Run::

    python -m drl_mpc_nav.cli.train --help
    python -m drl_mpc_nav.cli.train --variant-index 1 --total-timesteps 50000
"""
from __future__ import annotations

import logging
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import tyro
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv

from drl_alg.per_ddpg import PerDDPG
from drl_alg.utils.map import (
    generate_map_corridor,
    generate_map_dynamic,
    generate_map_eval,
    generate_map_mpc,
)
from drl_env import MapDescription
from drl_mpc_nav.config import TrainConfig

logger = logging.getLogger(__name__)


#: Pre-defined training variants.  Index matches ``TrainConfig.variant_index``.
VARIANTS = [
    {
        "algorithm": "DDPG",
        "env_name": "TrajectoryPlannerEnvironmentImgsReward1-v0",
        "net_arch": [64, 64],
        "per": True,
        "device": "auto",
    },
    {
        "algorithm": "DDPG",
        "env_name": "TrajectoryPlannerEnvironmentImgsReward2-v0",
        "net_arch": [64, 64],
        "per": True,
        "device": "auto",
    },
    {
        "algorithm": "DDPG",
        "env_name": "TrajectoryPlannerEnvironmentRaysReward1-v0",
        "net_arch": [16, 16],
        "per": True,
        "device": "cpu",
    },
    {
        "algorithm": "DDPG",
        "env_name": "TrajectoryPlannerEnvironmentRaysReward2-v0",
        "net_arch": [16, 16],
        "per": True,
        "device": "cpu",
    },
    {
        "algorithm": "DDPG",
        "env_name": "TrajectoryPlannerEnvironmentImgsReward-v0",
        "net_arch": [64, 64],
        "per": False,
        "device": "auto",
    },
    {
        "algorithm": "DDPG",
        "env_name": "TrajectoryPlannerEnvironmentRaysReward-v0",
        "net_arch": [16, 16],
        "per": False,
        "device": "cpu",
    },
]


def _make_map() -> MapDescription:
    import random as _random

    # generate_map_mpc() is a factory returning a MapGenerator; the others are
    # already MapGenerators — all three are callable and equivalent at this point.
    map_generators = [generate_map_dynamic, generate_map_corridor, generate_map_mpc()]
    return _random.choice(map_generators)()


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(cfg: TrainConfig) -> None:
    """Run training with the given configuration.

    Args:
        cfg: Fully populated :class:`~drl_mpc_nav.config.TrainConfig`.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    _set_seeds(cfg.seed)

    if cfg.variant_index >= len(VARIANTS):
        raise ValueError(
            f"variant_index must be in [0, {len(VARIANTS) - 1}], got {cfg.variant_index}"
        )

    variant = VARIANTS[cfg.variant_index]
    save_path = str(cfg.output_dir / f"variant-{cfg.variant_index}" / f"run{cfg.run_version}")

    logger.info("Training variant %d → %s", cfg.variant_index, save_path)

    vec_env = make_vec_env(
        variant["env_name"],
        n_envs=cfg.n_envs,
        seed=cfg.seed,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"generate_map": _make_map},
    )
    vec_env_eval = make_vec_env(
        variant["env_name"],
        n_envs=cfg.n_envs,
        seed=cfg.seed,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"generate_map": generate_map_eval},
    )

    n_actions = vec_env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions),
    )

    AlgorithmClass = PerDDPG if variant["per"] else DDPG

    model = AlgorithmClass(
        "MultiInputPolicy",
        vec_env,
        action_noise=action_noise,
        policy_kwargs={"net_arch": variant["net_arch"]},
        device=cfg.device,
        verbose=1,
    )

    if cfg.load_checkpoint:
        import os

        checkpoint = os.path.join(save_path, "best_model.zip")
        if os.path.exists(checkpoint):
            logger.info("Loading checkpoint from %s", checkpoint)
            model.set_parameters(checkpoint)
        else:
            logger.warning("Checkpoint not found at %s, starting fresh.", checkpoint)

    eval_callback = EvalCallback(
        vec_env_eval,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=max(int(cfg.total_timesteps / 100) // cfg.n_envs, 1),
        n_eval_episodes=cfg.n_envs,
    )

    model.learn(total_timesteps=int(cfg.total_timesteps), callback=eval_callback)
    logger.info("Training complete.  Best model saved to %s", save_path)


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
