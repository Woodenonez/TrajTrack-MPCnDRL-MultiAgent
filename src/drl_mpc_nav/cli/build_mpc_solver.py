"""CLI entry point for (re-)building the MPC solver.

Run::

    python -m drl_mpc_nav.cli.build_mpc_solver --help
    python -m drl_mpc_nav.cli.build_mpc_solver --config-file mpc_default.yaml
"""
from __future__ import annotations

import tyro

import motion_model
from drl_mpc_nav.config import MPCSolverBuildConfig
from drl_mpc_nav.utils.paths import config_path
from mpc_traj_tracker import MPCConfig, TrajectoryGenerator


def main(cfg: MPCSolverBuildConfig) -> None:
    """Build (or rebuild) the MPC solver for the given configuration.

    Args:
        cfg: Fully populated :class:`~drl_mpc_nav.config.MPCSolverBuildConfig`.
    """
    yaml_fp = config_path(cfg.config_file)
    mpc_cfg = MPCConfig.from_yaml(str(yaml_fp))

    if cfg.config_file == "mpc_default.yaml":
        input(
            "\033[91mThis will overwrite the default solver. "
            "Press Enter to continue or Ctrl-C to abort.\033[0m\n"
        )

    TrajectoryGenerator(
        mpc_cfg,
        motion_model.motion_model.unicycle_model,
        build_solver=True,
        use_tcp=cfg.use_tcp,
        verbose=cfg.verbose,
    )


if __name__ == "__main__":
    main(tyro.cli(MPCSolverBuildConfig))
