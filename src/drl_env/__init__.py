from .agent import MobileRobot
from .obstacle import Boundary, Obstacle, Animation
from .goal import Goal

from typing import Callable, Tuple, List

MapDescription = Tuple[MobileRobot, Boundary, List[Obstacle], Goal]
MapGenerator = Callable[[], MapDescription]

__all__ = ['MobileRobot', 'Boundary', 'Obstacle', 'Animation', 'Goal', 'MapDescription']

from gymnasium import register


max_episode_steps = 1000

register(
    id='TrajectoryPlannerEnvironmentImgsReward-v0',
    entry_point='drl_env.variants.imgs_reward:TrajectoryPlannerEnvironmentImgsReward',
    max_episode_steps=max_episode_steps,
)
register(
    id='TrajectoryPlannerEnvironmentRaysReward-v0',
    entry_point='drl_env.variants.rays_reward:TrajectoryPlannerEnvironmentRaysReward',
    max_episode_steps=max_episode_steps,
)
