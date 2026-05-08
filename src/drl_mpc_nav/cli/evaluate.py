"""CLI entry point for evaluation.

Run::

    python -m drl_mpc_nav.cli.evaluate --help
    python -m drl_mpc_nav.cli.evaluate --decision-mode hybrid --rl-index 0
"""
from __future__ import annotations

import copy
import logging
import random
import warnings

warnings.filterwarnings("ignore")

import gymnasium as gym
import numpy as np
import torch
import tyro
from stable_baselines3.common import env_checker
from torch import no_grad

from drl_alg.per_ddpg import PerDDPG
from drl_env import MobileRobot
from drl_env.environment import TrajectoryPlannerEnvironment
from drl_mpc_nav.config import DecisionMode, EvalConfig
from drl_mpc_nav.evaluation.metrics import Metrics
from drl_mpc_nav.hybrid.planner import circle_to_rect, est_dyn_obs_positions
from drl_mpc_nav.hybrid.ref_filter import ref_traj_filter
from drl_mpc_nav.hybrid.switching import HintSwitcher
from drl_mpc_nav.utils.paths import config_path, model_path
from drl_mpc_nav.utils.timing import PieceTimer
from helper import generate_map, get_geometric_map  # type: ignore[import]
from mpc_traj_tracker import MPCConfig, TrajectoryGenerator

logger = logging.getLogger(__name__)


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_rl_model_env(
    map_generator,
    index: int,
) -> tuple[PerDDPG, TrajectoryPlannerEnvironment]:
    """Load a pre-trained RL model and its corresponding environment.

    Args:
        map_generator: Callable that returns a ``MapDescription``.
        index:         0 = image-based DDPG, 1 = ray-based DDPG.

    Returns:
        Tuple of ``(model, unwrapped_env)``.

    Raises:
        ValueError: If *index* is not 0 or 1.
    """
    variants = [
        {
            "env_name": "TrajectoryPlannerEnvironmentImgsReward-v0",
            "net_arch": [64, 64],
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "model_subfolder": "image",
        },
        {
            "env_name": "TrajectoryPlannerEnvironmentRaysReward-v0",
            "net_arch": [16, 16],
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "model_subfolder": "ray",
        },
    ]
    if index not in (0, 1):
        raise ValueError(f"rl_index must be 0 or 1, got {index}")

    variant = variants[index]
    pt_path = model_path(f"ddpg/{variant['model_subfolder']}/best_model.pt")

    env: TrajectoryPlannerEnvironment = gym.make(
        variant["env_name"], generate_map=map_generator, discrete_action=False
    )
    env_checker.check_env(env)

    ddpg_model = PerDDPG(
        "MultiInputPolicy",
        env,
        policy_kwargs={"net_arch": variant["net_arch"]},
        device=variant["device"],
    )
    ddpg_model.policy.load_state_dict(
        torch.load(pt_path, map_location=variant["device"], weights_only=False)
    )
    return ddpg_model, env.unwrapped  # type: ignore[return-value]


def run_episode(
    cfg: EvalConfig,
    ddpg_model: PerDDPG,
    env_eval: TrajectoryPlannerEnvironment,
    traj_gen: TrajectoryGenerator,
    geo_map,
) -> tuple[list[float], bool, list[tuple[float, float]], list, list, list]:
    """Execute a single navigation episode.

    Returns:
        ``(time_list, success, action_list, ref_traj, traversed_positions, obstacle_list)``
    """
    time_list: list[float] = []
    dyn_obs_size = cfg.dyn_obs_size

    obsv, *_ = env_eval.reset()

    init_state = np.array([*env_eval.agent.position, env_eval.agent.angle])
    goal_state = np.array([*env_eval.goal.position, 0])
    ref_path = list(env_eval.path.coords)
    traj_gen.load_init_state(init_state, goal_state)
    traj_gen.set_work_mode(mode="work")
    traj_gen.set_ref_trajectory(ref_path)

    last_mpc_time = 0.0
    last_rl_time = 0.0
    chosen_ref_traj = None
    rl_ref = None
    last_dyn_obstacle_list = None

    switch = HintSwitcher(10, 2, 10)
    done = False

    for i in range(cfg.max_steps):
        logger.debug("Step %d/%d", i + 1, cfg.max_steps)

        dyn_obstacle_list = [
            obs.keyframe.position.tolist()
            for obs in env_eval.obstacles
            if not obs.is_static
        ]
        dyn_obstacle_list_poly = [circle_to_rect(obs, dyn_obs_size) for obs in dyn_obstacle_list]
        if last_dyn_obstacle_list is None:
            last_dyn_obstacle_list = dyn_obstacle_list
        dyn_obstacle_pred_list = [
            est_dyn_obs_positions(last_dyn_obstacle_list[j], obs, obs_size=dyn_obs_size)
            for j, obs in enumerate(dyn_obstacle_list)
        ]
        last_dyn_obstacle_list = dyn_obstacle_list

        if cfg.decision_mode == DecisionMode.MPC:
            env_eval.set_agent_state(
                traj_gen.state[:2], traj_gen.state[2],
                traj_gen.last_action[0], traj_gen.last_action[1],
            )
            obsv, _reward, done, _trunc, info = env_eval.step([0, 0])
            if dyn_obstacle_list:
                traj_gen.update_dynamic_constraints(dyn_obstacle_pred_list)
            original_ref_traj, *_ = traj_gen.get_local_ref_traj()
            chosen_ref_traj = original_ref_traj
            timer_mpc = PieceTimer()
            try:
                mpc_output = traj_gen.get_action(chosen_ref_traj)
            except Exception as exc:
                logger.warning("MPC failed: %s", exc)
                done = True
                break
            last_mpc_time = timer_mpc(4, ms=True)
            if mpc_output is None:
                break

        elif cfg.decision_mode == DecisionMode.DDPG:
            traj_gen.set_current_state(env_eval.agent.state)
            original_ref_traj, *_ = traj_gen.get_local_ref_traj()
            timer_rl = PieceTimer()
            action_index, _ = ddpg_model.predict(obsv, deterministic=True)
            last_rl_time = timer_rl(4, ms=True)
            obsv, _reward, done, _trunc, info = env_eval.step(action_index)

        elif cfg.decision_mode == DecisionMode.HYBRID:
            env_eval.set_agent_state(
                traj_gen.state[:2], traj_gen.state[2],
                traj_gen.last_action[0], traj_gen.last_action[1],
            )
            timer_rl = PieceTimer()
            action_index, _ = ddpg_model.predict(obsv, deterministic=True)
            obsv, _reward, done, _trunc, info = env_eval.step(action=None)

            rl_ref = []
            if cfg.new_rl_ref:
                env_sim: TrajectoryPlannerEnvironment = copy.deepcopy(env_eval)
                rl_skip = 0
                robot_sim: MobileRobot = copy.deepcopy(env_eval.agent)
                for _ in range(20):
                    robot_sim.step(action_index, traj_gen.config.ts)
                    rl_ref.append(list(robot_sim.position))
                    if rl_skip == 0:
                        env_sim.agent = robot_sim
                        env_sim.step_obstacles()
                        env_sim.update_status(reset=False)
                        obsv_sim = env_sim.get_observation()
                        action_index, _ = ddpg_model.predict(obsv_sim, deterministic=True)
                        rl_skip += 1
                    else:
                        rl_skip += 1
                        if rl_skip >= 3:
                            rl_skip = 0
            else:
                robot_sim = copy.deepcopy(env_eval.agent)
                for j in range(20):
                    if j == 0:
                        robot_sim.step(action_index, traj_gen.config.ts)
                    else:
                        robot_sim.step_with_decay_angular_velocity(traj_gen.config.ts, 1.0)
                    rl_ref.append(list(robot_sim.position))
            last_rl_time = timer_rl(4, ms=True)

            if dyn_obstacle_list:
                traj_gen.update_dynamic_constraints(dyn_obstacle_pred_list)
            original_ref_traj, rl_ref_traj, extra_ref_traj = traj_gen.get_local_ref_traj(
                np.array(rl_ref)
            )
            filtered_ref_traj = ref_traj_filter(original_ref_traj, rl_ref_traj, decay=1.0)
            all_obstacles = geo_map.processed_obstacle_list + dyn_obstacle_list_poly
            if switch.switch(traj_gen.state[:2], extra_ref_traj.tolist(),
                             filtered_ref_traj.tolist(), all_obstacles):
                chosen_ref_traj = filtered_ref_traj
            else:
                chosen_ref_traj = original_ref_traj
            timer_mpc = PieceTimer()
            try:
                mpc_output = traj_gen.get_action(chosen_ref_traj)
            except Exception as exc:
                logger.warning("MPC failed: %s", exc)
                done = True
                break
            last_mpc_time = timer_mpc(4, ms=True)

        else:
            raise ValueError(f"Unknown decision mode: {cfg.decision_mode}")

        # Accumulate timing
        if cfg.decision_mode == DecisionMode.MPC:
            time_list.append(last_mpc_time)
        elif cfg.decision_mode == DecisionMode.DDPG:
            time_list.append(last_rl_time)
        else:
            time_list.append(last_mpc_time + last_rl_time)

        if cfg.plot and i % 1 == 0:
            env_eval.render(dqn_ref=rl_ref, actual_ref=chosen_ref_traj)

        if i == cfg.max_steps - 1:
            done = True
            logger.info("Episode timed out.")
        if done:
            if cfg.plot:
                input(f"Finish (Succeed: {info['success']})! Press enter to continue...")
            break

    action_list = list(zip(env_eval.speeds, env_eval.angular_velocities, strict=False))
    return (
        time_list,
        info["success"],
        action_list,
        traj_gen.ref_traj,
        env_eval.traversed_positions,
        geo_map.obstacle_list,
    )


def main(cfg: EvalConfig) -> None:
    """Run evaluation with the given configuration.

    Args:
        cfg: Fully populated :class:`~drl_mpc_nav.config.EvalConfig`.
    """
    logging.basicConfig(
        level=logging.DEBUG if cfg.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    _set_seeds(cfg.seed)

    if cfg.verbose:
        logger.info("Decision mode: %s", cfg.decision_mode.value)

    map_gen = generate_map(*cfg.scene)

    ddpg_model, env_eval = load_rl_model_env(map_gen, cfg.rl_index)

    cfg_fpath = config_path(cfg.mpc_config.name) if not cfg.mpc_config.is_absolute() else cfg.mpc_config
    traj_gen = TrajectoryGenerator(MPCConfig.from_yaml(str(cfg_fpath)), motion_model=None)
    geo_map = get_geometric_map(env_eval.get_map_description(), inflate_margin=0.7)
    traj_gen.update_static_constraints(geo_map.processed_obstacle_list)

    num_trials = 1
    metrics = Metrics(mode=cfg.decision_mode.value.upper())

    with no_grad():
        for trial in range(num_trials):
            logger.info("Trial %d/%d", trial + 1, num_trials)
            time_list, success, actions, ref_traj, actual_traj, obstacle_list = run_episode(
                cfg, ddpg_model, env_eval, traj_gen, geo_map
            )
            metrics.add_trial_result(
                computation_time_list=time_list,
                succeed=success,
                action_list=actions,
                ref_trajectory=ref_traj,
                actual_trajectory=actual_traj,
                obstacle_list=obstacle_list,
            )

    avg = metrics.get_average(round_digit=3)
    print(f"\n=== Results ({cfg.decision_mode.value}) ===")
    print(avg)


if __name__ == "__main__":
    main(tyro.cli(EvalConfig))
