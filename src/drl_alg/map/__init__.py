from __future__ import annotations

from drl_env import MapDescription, MapGenerator

from .registry import (
    GENERATOR_REGISTRY,
    MODE_GENERATOR_NAMES,
    get_map_generator,
    choose_generator,
    choose_generator_for_mode,
    generate_for_mode,
)

from .generators import (
    generate_map_dynamic,
    generate_map_corridor,
    generate_map_eval,
    generate_map_mpc,
    generate_simple_map_dynamic,
    generate_simple_map_nonconvex,
    generate_simple_map_static,
    generate_map_multi_robot1,
    generate_map_multi_robot2,
    generate_map_multi_robot3,
    generate_map_multi_robot3_eval,
    generate_map_scene_1,
    generate_map_scene_2,
)

__all__ = [
    "MapDescription",
    "MapGenerator",
    "GENERATOR_REGISTRY",
    "MODE_GENERATOR_NAMES",
    "get_map_generator",
    "choose_generator",
    "choose_generator_for_mode",
    "generate_for_mode",
    "generate_map_dynamic",
    "generate_map_corridor",
    "generate_map_eval",
    "generate_map_mpc",
    "generate_simple_map_dynamic",
    "generate_simple_map_nonconvex",
    "generate_simple_map_static",
    "generate_map_multi_robot1",
    "generate_map_multi_robot2",
    "generate_map_multi_robot3",
    "generate_map_multi_robot3_eval",
    "generate_map_scene_1",
    "generate_map_scene_2",
]