import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))

from pathlib import Path
import os
import pathlib
from dataclasses import dataclass

import tyro # type: ignore
import numpy as np

import motion_model
from mpc_traj_tracker import MPCConfig, TrajectoryGenerator
from path_planning import LocalPathPlanner
from _scenario_simulator import RobotInfo, Simulator # type: ignore


@dataclass
class Args:
    """Run MPC trajectory tracking simulation.

    Args:
        config_file: Name of the YAML config file inside /config.
        build: Whether to rebuild the MPC solver.
        plot: Whether to plot the results in the loop.
        case_index: Index of the scenario to run. If None, give the hints.
    """
    config_file: str = "mpc_default.yaml"
    build: bool = False
    plot: bool = True
    case_index: int | None = 4 # if None, give the hints
    mpc_config: Path | None = None
    solver_directory: Path | None = None
    max_steps: int | None = None
    # show_animation = False
    # save_animation = False


def run_simulation(args: Args) -> dict[int | str, RobotInfo]:
    yaml_fp = os.path.join(pathlib.Path(__file__).resolve().parents[2], 'config', args.config_file)
    config = MPCConfig.from_yaml(str(args.mpc_config) if args.mpc_config is not None else yaml_fp)
    if args.solver_directory is not None:
        config.build_directory = str(args.solver_directory)

    if args.build:
        if args.config_file == 'mpc_default.yaml':
            input("\033[91mThis will overwrite the default config file. Press any key to continue OR Ctrl+C to stop.\033[0m")
        TrajectoryGenerator(config, motion_model.motion_model.unicycle_model, build_solver=True, use_tcp=False, verbose=True)

    ### Load simulator
    sim = Simulator(config, scene_index=args.case_index, inflate_margin=(config.vehicle_width+config.vehicle_margin))

    ### Local path
    lpp = LocalPathPlanner(sim.graph)

    ### Load robot
    color_list = ['b', 'r', 'g', 'y', 'c', 'm', 'k']
    for robot_id in range(len(sim.start)):
        start, end = sim.start[robot_id], sim.waypoints[robot_id][-1]
        ref_path = lpp.get_ref_path(start[:2], end[:2])
        sim.load_robot(robot_id, ref_path, np.array(start), np.array(end), mode='work', color=color_list[robot_id])

    ### Start & run MPC
    playback_dict = sim.run(sim.graph, sim.scanner, plot_in_loop=args.plot, max_steps=args.max_steps)

    ### Plot results (press any key to continue in dynamic mode if stuck)
    # from visualizer.mpc_plot import MpcPlotAfter
    # xx, xy     = np.array(state_list)[:,0],  np.array(state_list)[:,1]
    # uv, uomega = np.array(action_list)[:,0], np.array(action_list)[:,1]
    # plotter = MpcPlotAfter(config, legend_style='single', double_map=False)
    # plotter.plot_results(sim.graph, xx, xy, uv, uomega, cost_list, start, end, animation=show_animation, scanner=sim.scanner, video=save_animation)

    return playback_dict


def main(args: Args) -> None:
    run_simulation(args)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)