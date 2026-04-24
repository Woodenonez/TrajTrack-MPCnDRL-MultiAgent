# Multi-Agent Trajectory Planning and Tracking: Hybrid DRL and MPC

This repository contains a research codebase for collision-free trajectory planning and tracking of autonomous mobile robots by combining deep reinforcement learning (DRL) with model predictive control (MPC). The implementation covers both the earlier DQN-MPC formulation for single-robot navigation and the later DDPG-MPC formulation that extends the hybrid approach to continuous actions and multi-robot scenarios.

The project is grounded in the following two papers included in [doc](doc):

- [CASE2023-DQN-MPC](doc/CASE2023_DQN_MPC_Preprint.pdf), *Collision-Free Trajectory Planning of Mobile Robots by Integrating Deep Reinforcement Learning and Model Predictive Control*
- [IROS2024-DDPG-MPC-MultiAgent](doc/IROS2024_DDPG_MPC_MultiAgent_Preprint.pdf), *Bird’s-Eye-View Trajectory Planning of Multiple Robots using Continuous Deep Reinforcement Learning and Model Predictive Control*

## Research Summary

The main idea is to let DRL propose a local collision-avoidance trajectory and let MPC refine and track that trajectory while enforcing kinematic, dynamic, and obstacle-related constraints.

- In the 2023 paper, a DQN agent generates an alternative local reference when the original path is blocked.
- In the 2024 paper, the approach is extended with DDPG, continuous actions, Bird’s-Eye-View image input, and sequential multi-robot execution.
- MPC remains the final low-level decision maker, producing smooth, constraint-aware control actions.

This hybrid design targets the tradeoff between the strengths and weaknesses of the individual methods:

- DRL provides fast, fixed-cost inference and can suggest feasible paths around complex or non-convex obstacles.
- MPC improves smoothness, constraint handling, and trajectory tracking quality.
- The combined pipeline aims to outperform pure MPC and pure DRL in runtime robustness, path quality, and obstacle avoidance.

## Method Overview

At a high level, the workflow is:

1. A global or reference path is given.
2. The environment provides observations, either Bird’s-Eye-View images or ray-based sensing.
3. A DRL policy proposes a short-horizon alternative trajectory or action sequence.
4. The MPC trajectory tracker converts that proposal into feasible control commands while respecting robot dynamics, static obstacles, dynamic obstacles, and other robots.
5. In multi-robot settings, other robots are modeled as dynamic obstacles with predicted trajectories for the current robot.

The implementation supports three operating modes that are used throughout the tests and evaluation scripts:

- pure MPC
- pure RL
- hybrid RL + MPC

## Repository Structure

### Core source modules

- [src/drl_env](src/drl_env): Gymnasium environments, robot state handling, obstacles, goals, rendering, and registered RL environments.
- [src/drl_alg](src/drl_alg): RL algorithms and prioritized replay implementations, including custom DDPG and DQN variants.
- [src/mpc_traj_tracker](src/mpc_traj_tracker): MPC configuration, solver generation, solver loading, and trajectory tracking logic.

### Tests and experiment runners

- [test/test_mpc.py](test/test_mpc.py): MPC-only simulation and optional solver rebuild.
- [test/test_dqn.py](test/test_dqn.py): DQN, MPC, and hybrid DQN-MPC evaluation.
- [test/test_ddpg.py](test/test_ddpg.py): DDPG, MPC, and hybrid DDPG-MPC evaluation.

### Assets and configuration

- [config/mpc_default.yaml](config/mpc_default.yaml): default MPC parameters, constraints, horizon, and solver naming.
- [model](model): pretrained DQN and DDPG checkpoints for image-based and ray-based observation variants.

## Environment Variants

Two RL observation modalities are supported by the registered Gymnasium environments:

- `TrajectoryPlannerEnvironmentImgsReward-v0`: Bird’s-Eye-View image input
- `TrajectoryPlannerEnvironmentRaysReward-v0`: ray-based sensing input

In the research context, these correspond to two different perception assumptions:

- ceiling-mounted or Bird’s-Eye-View sensing
- onboard ray or lidar-style sensing

## Installation

### Prerequisites

- Python 3.10 or newer
- Rust toolchain with `cargo` available, for MPC solver generation and rebuilds
- A Linux environment is recommended for the current setup

We recommend using `UV` for Python environment management.
Install the Python package in editable mode:
```bash
uv pip install -e .
```

## MPC Solver Build

The MPC solver is generated through OpEn and compiled into Rust/Python bindings under [mpc_solver](mpc_solver). A prebuilt solver is already present in this repository, but you can rebuild it when changing MPC dynamics or configuration.

To rebuild the default solver, use the MPC test entry point with the build flag enabled:

```bash
python test/test_mpc.py --build True --plot False
```

The default solver name and build directory are defined in [config/mpc_default.yaml](config/mpc_default.yaml).

## Quick Start

### Run the MPC baseline

```bash
python test/test_mpc.py
```

This runs the MPC-only simulator using the default configuration and plotting enabled.

### Run the DDPG experiments

```bash
python test/test_ddpg.py
```

This script evaluates the DDPG setup and supports pure MPC, pure RL, and hybrid RL-MPC comparisons depending on how the script parameters are configured.

### Run the DQN experiments

```bash
python test/test_dqn.py
```

This reproduces the earlier DQN-based hybrid setup used in the precursor work.

### Run evaluation logic from source

```bash
python src/evaluation.py
```

This script contains the main evaluation loop used to compare pure MPC, pure DDPG, and hybrid DDPG-MPC in dynamic scenes.

### Train a policy locally

```bash
python src/continous_training_local.py
```

The local training script defines the training variants directly in code. It uses Stable-Baselines3 vectorized environments and can load an existing checkpoint or start a new run depending on the script settings.

### Train on a cluster

For SLURM-based training, see [doc/training/README.md](doc/training/README.md) and [doc/training/SLURM_jobscript.sh](doc/training/SLURM_jobscript.sh).

## Pretrained Models

Pretrained checkpoints are included under [model](model):

- DQN image and ray variants
- DDPG image and ray variants

Each model directory contains typical Stable-Baselines3 artifacts such as:

- `best_model.pt`
- `final_model.pt`
- `evaluations.npz`

These checkpoints are used by the evaluation scripts and test runners.

## Citation

If you use this repository in academic work, cite the corresponding paper for the experiment setup you build upon:

- CASE 2023 DQN-MPC paper for the discrete-action single-robot formulation
- IROS 2024 DDPG-MPC paper for the continuous-action and multi-robot formulation

## Status

This repository is a research codebase rather than a polished end-user package. Some experiment settings are configured directly in the source files, and training workflows are still script-centric. The included pretrained checkpoints and test scripts provide the most direct path for reproducing results and exploring the method.