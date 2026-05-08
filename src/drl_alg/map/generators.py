from __future__ import annotations

from drl_env import MapDescription, MapGenerator

from .map_base import (
    generate_map_mpc,
    generate_map_dynamic,
    generate_map_corridor,
    generate_map_scene_1,
    generate_map_scene_2,
    generate_map_eval,
    generate_map_eval_rand,
)
from .map_extra import (
    generate_simple_map_dynamic,
    generate_simple_map_static,
    generate_simple_map_nonconvex,
)
from .map_multi_robot import (
    generate_map_multi_robot1,
    generate_map_multi_robot2,
    generate_map_multi_robot3,
    generate_map_multi_robot3_eval,
)

def mpc_random() -> MapDescription:
    return generate_map_mpc()()


def mpc_11() -> MapDescription:
    return generate_map_mpc(11)()