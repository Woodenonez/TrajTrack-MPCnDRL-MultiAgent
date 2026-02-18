### System import
from typing import List, Tuple
import os
import copy
import pathlib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

### DRL import
import torch
from torch import no_grad
import gymnasium as gym
from stable_baselines3.common import env_checker
from drl_ddpg.per_ddpg import PerDDPG

from drl_env import MobileRobot
from drl_env.environment import TrajectoryPlannerEnvironment

### MPC import
from mpc_traj_tracker import MPCConfig, TrajectoryGenerator

### Helper
from helper import generate_map, get_geometric_map, HintSwitcher, PieceTimer


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

def load_rl_model_env(generate_map, index: int) -> Tuple[PerDDPG, TrajectoryPlannerEnvironment]:
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
    
    env_eval:TrajectoryPlannerEnvironment = gym.make(variant['env_name'], generate_map=generate_map, discrete_action=False)
    env_checker.check_env(env_eval)
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


def main(rl_index:int=1, decision_mode:int=1, to_plot=False, scene_option:Tuple[int, int, int]=(1, 1, 1), save_num:int=1):
    """
    Args:
        rl_index: 0 for image, 1 for ray
        decision_mode: 0 for pure mpc, 1 for pure ddpg, 2 for hybrid
    """
    prt_decision_mode = {0: 'pure_mpc', 1: 'pure_ddpg', 2: 'hybrid_ddpg'}
    print(f"The decision mode is: {prt_decision_mode[decision_mode]}")

    time_list = []

    ddpg_model, env_eval = load_rl_model_env(generate_map(*scene_option), rl_index)
    env_eval: TrajectoryPlannerEnvironment = env_eval.unwrapped

    CONFIG_FN = 'mpc_default.yaml'
    cfg_fpath = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'config', CONFIG_FN)
    traj_gen = TrajectoryGenerator(MPCConfig.from_yaml(cfg_fpath), motion_model=None)
    geo_map = get_geometric_map(env_eval.get_map_description(), inflate_margin=0.8)
    traj_gen.update_static_constraints(geo_map.processed_obstacle_list) # if assuming static obstacles not changed

    done = False
    with no_grad():
        while not done:
            obsv, *_ = env_eval.reset()

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

            for i in range(0, MAX_RUN_STEP):
                
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
                    obsv, reward, done, truncated, info = env_eval.step([0,0]) # just for plotting and updating status

                    if dyn_obstacle_list:
                        traj_gen.update_dynamic_constraints(dyn_obstacle_pred_list)
                    original_ref_traj, *_ = traj_gen.get_local_ref_traj()
                    chosen_ref_traj = original_ref_traj
                    timer_mpc = PieceTimer()
                    try:
                        mpc_output = traj_gen.get_action(chosen_ref_traj)
                    except Exception as e:
                        done = True
                        print(f'MPC fails: {e}')
                        break
                    last_mpc_time = timer_mpc(4, ms=True)
                    if mpc_output is None:
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
                    if NEW_RL_REF:
                        env_sim:TrajectoryPlannerEnvironment = copy.deepcopy(env_eval)
                        rl_skip = 0
                        for j in range(20):
                            env_sim.step_agent(action_index)
                            rl_ref.append(list(env_sim.agent.position))
                            if rl_skip == 0:
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
                        robot_sim:MobileRobot = copy.deepcopy(env_eval.agent)
                        robot_sim:MobileRobot
                        for j in range(20):
                            if j == 0:
                                robot_sim.step(action_index, traj_gen.config.ts)
                            else:
                                robot_sim.step_with_decay_angular_velocity(traj_gen.config.ts, 1.0)
                            rl_ref.append(list(robot_sim.position))
                    last_rl_time = timer_rl(4, ms=True)
                    # last_rl_ref = rl_ref
                    
                    if dyn_obstacle_list:
                        # traj_gen.update_dynamic_constraints([dyn_obstacle_tmp*20])
                        traj_gen.update_dynamic_constraints(dyn_obstacle_pred_list)
                    original_ref_traj, rl_ref_traj, extra_ref_traj = traj_gen.get_local_ref_traj(np.array(rl_ref))
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


                if to_plot & (i%1==0): # render
                    env_eval.render(dqn_ref=rl_ref, actual_ref=chosen_ref_traj, original_ref=original_ref_traj, save=False, save_num=save_num)

                if i == MAX_RUN_STEP - 1:
                    done = True
                    print('Time out!')
                if done:
                    if to_plot:
                        input(f"Finish (Succeed: {info['success']})! Press enter to continue...")
                    break

    print(f"Average time ({prt_decision_mode[decision_mode]}): {np.mean(time_list)}ms\n")
    return time_list

if __name__ == '__main__':
    """
    test_scene_1_dict = {1: [1, 2, 3], 2: [1, 2, 3, 4], 3: [1, 2, 3, 4], 4: [1, 2]}
    test_scene_2_dict = {1: [1, 2, 3]}


    decision_mode: 0 = MPC, 1 = DDPG, 2 = TD3, 3 = Hybrid DDPG, 4 = Hybrid TD3  
    """
    scene_option = (1, 3, 2)

    # time_list_mpc     = main(rl_index=1,    decision_mode=0,  to_plot=False, scene_option=scene_option, save_num=1) # Eval MPC using main.py
    # time_list_img     = main(rl_index=0,    decision_mode=1,  to_plot=False, scene_option=scene_option, save_num=3)
    time_list_hyb_img = main(rl_index=0,    decision_mode=2,  to_plot=True, scene_option=scene_option, save_num=5)

    input('Press enter to exit...')