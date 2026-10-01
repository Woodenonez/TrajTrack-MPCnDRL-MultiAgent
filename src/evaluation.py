from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from util.run_records import EpisodeRecord

### System import
import os
import copy
import pathlib
import warnings
warnings.filterwarnings("ignore")

import numpy as np

### DRL import
import torch
from torch import no_grad
import gymnasium as gym
from stable_baselines3.common import env_checker
from util.env_validation import check_env_isolated
from drl_alg.per_ddpg import PerDDPG

from drl_env import MobileRobot
from drl_env.environment import TrajectoryPlannerEnvironment

### MPC import
from mpc_traj_tracker import MPCConfig, TrajectoryGenerator

### Helper
from helper import generate_map, get_geometric_map, HintSwitcher, PieceTimer, Metrics


MAX_RUN_STEP = 200
DYN_OBS_SIZE = 0.8 + 0.8
NEW_RL_REF = True


def ref_traj_filter(original: np.ndarray, new: np.ndarray, decay=1):
    filtered = original.copy()
    for i in range(filtered.shape[0]):
        filtered[i, :] = (1-decay) * filtered[i, :] + decay * new[i, :]
        decay *= decay
        if decay < 1e-2:
            decay = 0.0
    return filtered

def load_rl_model_env(generate_map, index: int, *, checkpoint: Path | None = None) -> tuple[PerDDPG, TrajectoryPlannerEnvironment]:
    variant = [
        {
            'env_name': 'TrajectoryPlannerEnvironmentImgsReward-v0',
            'net_arch': [64, 64],
            'per': False,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        },
        {
            'env_name': 'TrajectoryPlannerEnvironmentRaysReward-v0',
            'net_arch': [16, 16],
            'per': False,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        },
    ][index] 

    if index == 0:
        model_folder_name = 'image'
    elif index == 1:
        model_folder_name = 'ray'
    else:
        raise ValueError('Invalid index')
    model_path = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'model/ddpg', model_folder_name, 'best_model.pt')
    
    if checkpoint is not None:
        model_path = str(checkpoint)

    env_eval:TrajectoryPlannerEnvironment = gym.make(variant['env_name'], generate_map=generate_map, discrete_action=False)
    check_env_isolated(env_eval, env_checker.check_env)
    ddpg_model = PerDDPG("MultiInputPolicy", env_eval, policy_kwargs={'net_arch': variant['net_arch']}, device=variant['device'])
    ddpg_model.policy.load_state_dict(torch.load(model_path, map_location=variant['device'], weights_only=False))
    return ddpg_model, env_eval

def est_dyn_obs_positions(last_pos: list, current_pos: list, steps:int=20):
    """
    Estimate the dynamic obstacle positions in the future.
    """
    est_pos = []
    d_pos = [current_pos[0]-last_pos[0], current_pos[1]-last_pos[1]]
    for i in range(steps):
        est_pos.append([current_pos[0]+d_pos[0]*(i+1), current_pos[1]+d_pos[1]*(i+1), DYN_OBS_SIZE, DYN_OBS_SIZE, 0, 1])
    return est_pos

def circle_to_rect(pos: list, radius:float=DYN_OBS_SIZE):
    """Convert the circle to a rectangle."""
    return [[pos[0]-radius, pos[1]-radius], [pos[0]+radius, pos[1]-radius], [pos[0]+radius, pos[1]+radius], [pos[0]-radius, pos[1]+radius]]


def main_process(rl_index:int=1, decision_mode:int=1, to_plot=False, scene_option:tuple[int, int, int]=(1, 1, 1), verbose:bool=False, *, checkpoint: Path | None = None, mpc_config: Path | None = None, solver_directory: Path | None = None, max_steps: int | None = None, record: EpisodeRecord | None = None):
    """
    Args:
        rl_index: 0 for image, 1 for ray
        decision_mode: 0 for pure mpc, 1 for pure ddpg, 2 for hybrid
    """
    if verbose:
        prt_decision_mode = {0: 'pure_mpc', 1: 'pure_ddpg', 2: 'hybrid'}
        print(f"The decision mode is: {prt_decision_mode[decision_mode]}")

    step_limit: int = MAX_RUN_STEP if max_steps is None else max_steps
    if step_limit <= 0:
        raise ValueError("max_steps must be positive")

    time_list = []

    ddpg_model, env_eval = load_rl_model_env(generate_map(*scene_option), rl_index, checkpoint=checkpoint)
    env_eval: TrajectoryPlannerEnvironment = env_eval.unwrapped

    CONFIG_FN = 'mpc_default.yaml'
    cfg_fpath = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'config', CONFIG_FN)
    config = MPCConfig.from_yaml(str(mpc_config) if mpc_config is not None else cfg_fpath)
    if solver_directory is not None:
        config.build_directory = str(solver_directory)
    traj_gen = TrajectoryGenerator(config, motion_model=None)
    geo_map = get_geometric_map(env_eval.get_map_description(), inflate_margin=0.7)
    traj_gen.update_static_constraints(geo_map.processed_obstacle_list) # assuming static obstacles not changed

    done = False
    with no_grad():
        while not done:
            obsv, *_  = env_eval.reset()

            init_state = np.array([*env_eval.agent.position, env_eval.agent.angle])
            goal_state = np.array([*env_eval.goal.position, 0])
            ref_path = list(env_eval.path.coords)
            traj_gen.load_init_state(init_state, goal_state)
            traj_gen.set_work_mode(mode='work')
            traj_gen.set_ref_trajectory(ref_path)

            last_mpc_time = 0.0
            last_rl_time = 0.0

            chosen_ref_traj = None
            rl_ref = None  
            last_dyn_obstacle_list = None            

            switch = HintSwitcher(10, 2, 10)

            for i in range(0, step_limit):

                print(f"\r{decision_mode}, {i+1}/{step_limit}", end="  ")

                dyn_obstacle_list = [obs.keyframe.position.tolist() for obs in env_eval.obstacles if not obs.is_static]
                dyn_obstacle_tmp  = [obs+[DYN_OBS_SIZE, DYN_OBS_SIZE, 0, 1] for obs in dyn_obstacle_list]
                dyn_obstacle_list_poly = [circle_to_rect(obs) for obs in dyn_obstacle_list]
                dyn_obstacle_pred_list = []
                if last_dyn_obstacle_list is None:
                    last_dyn_obstacle_list = dyn_obstacle_list
                for j, dyn_obs in enumerate(dyn_obstacle_list):
                    dyn_obstacle_pred_list.append(est_dyn_obs_positions(last_dyn_obstacle_list[j], dyn_obs))
                last_dyn_obstacle_list = dyn_obstacle_list

                if decision_mode == 0:
                    env_eval.set_agent_state(traj_gen.state[:2], traj_gen.state[2], 
                                                traj_gen.last_action[0], traj_gen.last_action[1])
                    obsv, reward, done, truncated, info = env_eval.step([0, 0]) # just for plotting and updating status

                    if dyn_obstacle_list:
                        traj_gen.update_dynamic_constraints(dyn_obstacle_pred_list)
                    original_ref_traj, *_ = traj_gen.get_local_ref_traj()
                    chosen_ref_traj = original_ref_traj
                    timer_mpc = PieceTimer()
                    try:
                        mpc_output = traj_gen.get_action(chosen_ref_traj)
                    except Exception as e:
                        done = True
                        if record is not None:
                            record.termination = 'solver_error'
                        print(f'MPC fails: {e}')
                        break
                    last_mpc_time = timer_mpc(4, ms=True)
                    if mpc_output is None:
                        done = True
                        if record is not None:
                            record.success = bool(info['success'])
                            record.collided = bool(env_eval.collided)
                            record.budget_exhausted = False
                            record.termination = 'completed'
                        break
                    action, pred_states, cost = mpc_output

                elif decision_mode == 1:
                    traj_gen.set_current_state(env_eval.agent.state)
                    original_ref_traj, *_ = traj_gen.get_local_ref_traj() # just for output

                    timer_rl = PieceTimer()
                    action_index, _states = ddpg_model.predict(obsv, deterministic=True)
                    last_rl_time = timer_rl(4, ms=True)
                    obsv, reward, done, truncated, info = env_eval.step(action_index)

                elif decision_mode == 2:
                    env_eval.set_agent_state(traj_gen.state[:2], traj_gen.state[2], 
                                             traj_gen.last_action[0], traj_gen.last_action[1])
                    timer_rl = PieceTimer()
                    action_index, _states = ddpg_model.predict(obsv, deterministic=True)
                    obsv, reward, done, truncated, info = env_eval.step(action=None) # step the environment but not the agent

                    rl_ref = []
                    robot_sim:MobileRobot = copy.deepcopy(env_eval.agent)
                    robot_sim:MobileRobot
                    if NEW_RL_REF:
                        env_sim:TrajectoryPlannerEnvironment = copy.deepcopy(env_eval)
                        rl_skip = 0
                        for j in range(20):
                            robot_sim.step(action_index, traj_gen.config.ts)
                            rl_ref.append(list(robot_sim.position))
                            if rl_skip == 0:
                                env_sim.agent = robot_sim
                                env_sim.step_obstacles()
                                env_sim.update_status(reset=False)
                                obsv_sim = env_sim.get_observation()
                                action_index, _states = ddpg_model.predict(obsv_sim, deterministic=True)
                                rl_skip += 1
                            else:
                                rl_skip += 1
                                if rl_skip >= 3:
                                    rl_skip = 0

                    else:
                        for j in range(20):
                            if j == 0:
                                robot_sim.step(action_index, traj_gen.config.ts)
                            else:
                                robot_sim.step_with_decay_angular_velocity(traj_gen.config.ts, 1.0)
                            rl_ref.append(list(robot_sim.position))
                    last_rl_time = timer_rl(4, ms=True)
                    
                    if dyn_obstacle_list:
                        traj_gen.update_dynamic_constraints(dyn_obstacle_pred_list)
                    original_ref_traj, rl_ref_traj,extra_ref_traj = traj_gen.get_local_ref_traj(np.array(rl_ref))
                    filtered_ref_traj = ref_traj_filter(original_ref_traj, rl_ref_traj, decay=1) # decay=1 means no decay
                    if switch.switch(traj_gen.state[:2], extra_ref_traj.tolist(), filtered_ref_traj.tolist(), geo_map.processed_obstacle_list+dyn_obstacle_list_poly):
                        chosen_ref_traj = filtered_ref_traj
                    else:
                        chosen_ref_traj = original_ref_traj
                    timer_mpc = PieceTimer()
                    try:
                        mpc_output = traj_gen.get_action(chosen_ref_traj) # MPC computes the action
                    except Exception as e:
                        done = True
                        if record is not None:
                            record.termination = 'solver_error'
                        print(f'MPC fails: {e}')
                        break
                    last_mpc_time = timer_mpc(4, ms=True)

                else:
                    raise ValueError("Invalid decision mode")
                
                if decision_mode == 0:
                    time_list.append(last_mpc_time)
                    if to_plot:
                        print(f"Step {i}.Runtime (MPC): {last_mpc_time}ms")
                elif decision_mode == 1:
                    time_list.append(last_rl_time)
                    if to_plot:
                        print(f"Step {i}.Runtime (DDPG): {last_rl_time}ms")
                elif decision_mode == 2:
                    time_list.append(last_mpc_time+last_rl_time)
                    if to_plot:
                        print(f"Step {i}.Runtime (Hybrid DDPG): {last_mpc_time+last_rl_time} = {last_mpc_time}+{last_rl_time}ms")


                if record is not None:
                    record.environment_states.append(env_eval.agent.state.copy())
                    if decision_mode == 1:
                        record.rl_actions.append(np.asarray(action_index).copy())
                    else:
                        record.mpc_states.append(traj_gen.state.copy())
                        if mpc_output is not None:
                            record.mpc_actions.append(np.asarray(mpc_output[0]).copy())
                    if chosen_ref_traj is not None:
                        record.chosen_references.append(np.asarray(chosen_ref_traj).copy())
                    record.timing_ms.append(float(time_list[-1]))
                    record.success = bool(info['success'])
                    record.collided = bool(env_eval.collided)
                    record.budget_exhausted = i == step_limit - 1 and not done
                    record.termination = ('success' if record.success else
                                          'collision' if record.collided else
                                          'step_limit' if record.budget_exhausted else
                                          'terminated' if done else 'running')

                if to_plot & (i%1==0): # render every third frame
                    env_eval.render(dqn_ref=rl_ref, actual_ref=chosen_ref_traj)

                if i == step_limit - 1:
                    done = True
                    if verbose:
                        print('Time out!')
                if done:
                    if to_plot:
                        input(f"Finish (Succeed: {info['success']})! Press enter to continue...")
                    break

    action_list = [(v, w) for (v, w) in zip(env_eval.speeds, env_eval.angular_velocities)]

    if verbose:
        print(f"Average time ({prt_decision_mode[decision_mode]}): " +
          (f"{np.mean(time_list)}ms\n" if time_list else "unavailable (no control steps)\n"))
    else:
        print()
    return time_list, info["success"], action_list, traj_gen.ref_traj, env_eval.traversed_positions, geo_map.obstacle_list

def main_evaluate(rl_index: int, decision_mode, metrics: Metrics, scene_option:tuple[int, int, int]) -> Metrics:
    to_plot = False
    time_list, success, actions, ref_traj, actual_traj, obstacle_list = main_process(rl_index=rl_index, decision_mode=decision_mode, to_plot=to_plot, scene_option=scene_option)
    metrics.add_trial_result(computation_time_list=time_list, succeed=success, action_list=actions, 
                             ref_trajectory=ref_traj, actual_trajectory=actual_traj, obstacle_list=obstacle_list)
    return metrics


if __name__ == '__main__':
    """
    rl_index: 0: image, 1: ray
    decision_mode: 0: mpc, 1: ddpg, 2: td3, 3: hybrid-ddpg

    Map:
    SCENE 1:
    - 1: Single rectangular static obstacle 
        - (1-small, 2-medium, 3-large)
    - 2: Two rectangular static obstacles 
        - (1-small stagger, 2-large stagger, 3-close aligned, 4-far aligned)
    - 3: Single non-convex static obstacle
        - (1-big u-shape, 2-small u-shape, 3-big v-shape, 4-small v-shape)
    - 4: Single dynamic obstacle
        - (1-crash, 2-cross)

    SCENE 2:
    - 1: Single rectangular obstacle
        - (1-right, 2-sharp, 3-u-shape)
    - 2: Single dynamic obstacle
        - (1-right, 2-sharp, 3-u-shape)

    rl_index: 0 = image, 1 = ray
    decision_mode: 0 = MPC, 1 = DDPG, 2 = TD3, 3 = Hybrid DDPG, 4 = Hybrid TD3  
    """
    num_trials = 5 # 50
    print_latex = False
    scene_option_list = [
                        #  (1, 1, 2), # a-medium
                        #  (1, 1, 3), # b-large
                        #  (1, 2, 1), # c-small
                        #  (1, 2, 2), # d-large
                        #  (1, 3, 1), # e-small
                         (1, 3, 2), # f-large
                        #  (1, 4, 1), # face-to-face
                        #  (2, 1, 1), # right turn with an obstacle
                        #  (2, 1, 2), # sharp turn with an obstacle
                        #  (2, 1, 3), # u-turn with an obstacle
                         ]
    

    for scene_option in scene_option_list:

        print(f"=== Scene {scene_option[0]}-{scene_option[1]}-{scene_option[2]} ===")

        mpc_metrics = Metrics(mode='MPC')
        ddpg_lid_metrics = Metrics(mode='DDPG-L')
        ddpg_img_metrics = Metrics(mode='DDPG-V')
        hyb_ddpg_lid_metrics = Metrics(mode='HYB-DDPG-L')
        hyb_ddpg_img_metrics = Metrics(mode='HYB-DDPG-V')

        for i in range(num_trials):
            print(f"Trial {i+1}/{num_trials}")
            mpc_metrics = main_evaluate(rl_index=1, decision_mode=0, metrics=mpc_metrics, scene_option=scene_option)
            # ddpg_lid_metrics = main_evaluate(rl_index=1, decision_mode=1, metrics=ddpg_lid_metrics, scene_option=scene_option)
            ddpg_img_metrics = main_evaluate(rl_index=0, decision_mode=1, metrics=ddpg_img_metrics, scene_option=scene_option)
            # hyb_ddpg_lid_metrics = main_evaluate(rl_index=1, decision_mode=2, metrics=hyb_ddpg_lid_metrics, scene_option=scene_option)
            hyb_ddpg_img_metrics = main_evaluate(rl_index=0, decision_mode=2, metrics=hyb_ddpg_img_metrics, scene_option=scene_option)

        round_digits = 3
        print(f"=== Scene {scene_option[0]}-{scene_option[1]}-{scene_option[2]} ===")
        print('MPC')
        print(mpc_metrics.get_average(round_digits), '\n')
        # print('DDPG Lidar')
        # print(ddpg_lid_metrics.get_average(round_digits), '\n')
        print('DDPG Image')
        print(ddpg_img_metrics.get_average(round_digits), '\n')
        # print('DDPG hybrid Lidar')
        # print(hyb_ddpg_lid_metrics.get_average(round_digits), '\n')
        print('DDPG hybrid Image')
        print(hyb_ddpg_img_metrics.get_average(round_digits))
        print('='*50)


        ## Write to latex
        if print_latex:
            print(f"=== Scene {scene_option[0]}-{scene_option[1]}-{scene_option[2]} ===")
            print(mpc_metrics.write_latex(round_digits))
            # print(ddpg_lid_metrics.write_latex(round_digits))
            print(ddpg_img_metrics.write_latex(round_digits))
            # print(hyb_ddpg_lid_metrics.write_latex(round_digits))
            print(hyb_ddpg_img_metrics.write_latex(round_digits))