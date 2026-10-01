# Trajectory planning with DRL and MPC

This research codebase runs single-robot DQN/DDPG, MPC, and hybrid comparisons, plus a separate fleet MPC simulator. The hybrid runners use one robot; the fleet runner uses sequential MPC with peer predictions. A generated native MPC solver is currently required for all experiment modes, including pure RL and is **not included** in this checkout.

## Setup

Use Python 3.11 and a Rust toolchain for solver builds. From the repository root:

```bash
uv sync --locked
source .venv/bin/activate
```

<details><summary>Conda alternative</summary>

```bash
conda env create -f environment.yml
conda activate trajtrack
python -m pip install -r requirements-conda.txt
python -m pip install --no-deps -e .
```

The exported requirements come from the same uv lock. Conda and cluster execution have not been verified here.
</details>

## Commands

| Need | Command |
| --- | --- |
| Generate the native solver | `python scripts/build_solver.py --build-directory mpc_solver/candidate` |
| Fleet MPC, case 4 | `python scripts/run_experiment.py --workflow mpc-demo --solver-directory mpc_solver/candidate` |
| Image DDPG hybrid | `python scripts/run_experiment.py --workflow ddpg-demo --decision hybrid --observation image --checkpoint pretrained_model/ddpg/image/best_model.pt --solver-directory mpc_solver/candidate` |
| Ray DQN hybrid | `python scripts/run_experiment.py --workflow dqn-demo --decision hybrid --observation ray --checkpoint pretrained_model/dqn/ray/best_model.pt --solver-directory mpc_solver/candidate` |
| Quantitative DDPG comparison | `python scripts/run_experiment.py --workflow ddpg-eval --solver-directory mpc_solver/candidate` |
| Ray/CPU training | `python scripts/train.py --mode local --index 1 --no-evaluation` |

Use `--visualization off` for a headless experiment, `--checkpoint PATH` for an explicit demo policy, and `--output-dir runs/NAME` for optional run records. The builder writes a candidate directory; pass that same directory to the launcher. The included `pretrained_model/{ddpg,dqn}/{image,ray}/best_model.pt` files are **policy state dictionaries** for demos. With `--path PATH`, training evaluation loads a full Stable-Baselines3 archive at `PATH/best_model`; do not pass a `.pt` demo policy there. `Model/` holds new training outputs and is separate from `pretrained_model/`.

`python scripts/train.py` defaults to training; the supplied `config/training.yaml` sets `evaluation: true`. Override it with `--no-evaluation` when training from that YAML. Training variants are image/CUDA index 0 and ray/CPU index 1. See [training details](doc/training/README.md).

Legacy demo modules remain under `tests/demos/`; use `scripts/run_experiment.py` for shipped checkpoints and explicit solver paths. `src/evaluation.py` remains the legacy quantitative entry. `src/mpc_traj_tracker/mpc_interface.py` is obsolete and is not used by these commands. Reported finish time is a step count; metric definitions are unchanged.

Lightweight checks: `python -m unittest discover -s tests -p 'test_launch.py'` and `python scripts/smoke.py --help`. A help response does not verify solver, checkpoint, or numerical execution.

The `drl_mpc_nav.cli.train`, `.evaluate`, and `.build_mpc_solver` module commands delegate to the corresponding scripts and use their arguments. The obsolete six-variant training interface is retired; use the two documented training variants. Run the automated suite with `python -m pytest` after the locked development install.

See [run records and future project integration](doc/run_records.md) for array meanings and the relationship to `DyObAv-MPCnEBM-Warehouse`.

<details><summary>Research references</summary>

- [CASE 2023 DQN–MPC preprint](doc/CASE2023_DQN_MPC_Preprint.pdf)
- [IROS 2024 DDPG–MPC preprint](doc/IROS2024_DDPG_MPC_MultiAgent_Preprint.pdf)

The papers describe historical experiments; this README documents the current checkout's runnable interfaces.
</details>
