"""Validate fixed-scene experiment environments without moving their robots."""
from __future__ import annotations

import copy
from collections.abc import Callable

import gymnasium as gym


def check_env_isolated(env: gym.Env, checker: Callable[[gym.Env], None]) -> None:
    """Check an independent scene, then initialize the real experiment once.

    Deepcopy alone retains the original map-generator closure. Freeze its map
    and give the checked environment a fresh copy on every reset, so sampled
    checker actions cannot corrupt either subsequent checks or the experiment.
    The experiment retains its original factory and episode stepping semantics.
    """
    scene = copy.deepcopy(env.unwrapped.generate_map())
    checked = copy.deepcopy(env)
    checked.unwrapped.generate_map = lambda: copy.deepcopy(scene)
    try:
        checker(checked)
    finally:
        checked.close()
    env.reset()
