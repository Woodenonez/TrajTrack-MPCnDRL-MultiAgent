"""Map generator registry and selection helpers."""

from __future__ import annotations

import random
from collections.abc import Sequence

from drl_env import MapDescription, MapGenerator

from . import generators


GENERATOR_REGISTRY: dict[str, MapGenerator] = {
    "dynamic": generators.generate_map_dynamic,
    "corridor": generators.generate_map_corridor,
    "eval_default": generators.generate_map_eval,
    "mpc_random": generators.mpc_random,
    "mpc_11": generators.mpc_11,
    "simple_dynamic": generators.generate_simple_map_dynamic,
    "simple_nonconvex": generators.generate_simple_map_nonconvex,
    "simple_static": generators.generate_simple_map_static,
    "multi_robot1": generators.generate_map_multi_robot1,
    "multi_robot2": generators.generate_map_multi_robot2,
    "multi_robot3": generators.generate_map_multi_robot3,
    "multi_robot3_eval": generators.generate_map_multi_robot3_eval,
}

MODE_GENERATOR_NAMES: dict[str, tuple[str, ...]] = {
    "local": ("dynamic", "corridor", "mpc_random"),
    "cluster": ("dynamic", "simple_dynamic", "simple_nonconvex", "simple_static"),
}


def get_map_generator(name: str) -> MapGenerator:
    try:
        return GENERATOR_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(GENERATOR_REGISTRY.keys()))
        raise ValueError(f"Unknown map generator '{name}'. Available: {available}") from exc


def choose_generator(names: Sequence[str]) -> MapGenerator:
    if not names:
        raise ValueError("names must not be empty")
    return get_map_generator(random.choice(list(names)))


def choose_generator_for_mode(mode: str) -> MapGenerator:
    try:
        names = MODE_GENERATOR_NAMES[mode]
    except KeyError as exc:
        available = ", ".join(sorted(MODE_GENERATOR_NAMES.keys()))
        raise ValueError(f"Unknown mode '{mode}'. Available: {available}") from exc
    return choose_generator(names)


def generate_for_mode(mode: str) -> MapDescription:
    return choose_generator_for_mode(mode)()
