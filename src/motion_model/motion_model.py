from typing import Union, Callable
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import casadi.casadi as cs # type: ignore


"""
A motion model is either holonomic or non-holonomic.
A holonomic model is a model where the change of the state is a function of the action only.
A non-holonomic model is a model where the change of the state is a function of the action and the state.

Supported holonomic (additive) models:
    - Omnidirectional model
Supported non-holonomic models:
    - Unicycle model
Ongoing:
    - Simple car model
Wishlist:
    - Ackermann model
"""


class MotionModelType(Enum):
    """The type of a motion model."""
    HOLONOMIC = 0
    NON_HOLONOMIC = 1
    PRESET = 2
    UNKNOWN = 3

class MotionModel(ABC):
    """An interface for a motion model under `numpy`.
    `next_s = f(s, a, ts)`
    
    Properties:
        motion_model_type: The type of the motion model.
        
    Methods:
        __call__: The motion model.
        zero_state: Return the zero state of the motion model.
        zero_action: Return the zero action of the motion model.
    """
    _motion_model_type: MotionModelType

    def __init__(self, model: Callable, state_dim: int, action_dim: int, sampling_time: float) -> None:
        self.model = model
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ts = sampling_time

    def __call__(self, state: np.ndarray, action: np.ndarray, ts: None | float = None, **kwargs) -> np.ndarray:
        """The sampling time can be changed at runtime."""
        if ts is not None:
            self.ts = ts
        return self.model(state, action, self.ts, **kwargs)

    @property
    def motion_model_type(self) -> MotionModelType:
        return self._motion_model_type
    
    def f(self, state: np.ndarray, action: np.ndarray, ts: None | float = None, **kwargs) -> np.ndarray:
        """The call of the motion model."""
        return self(state, action, ts, **kwargs)
    
    def f_noise(self, state: np.ndarray, action: np.ndarray, ts: None | float = None, noise_scale:float=0.01, **kwargs) -> np.ndarray:
        """The call of the motion model with noise."""
        return self(state, action, ts, **kwargs) + np.random.normal(0, noise_scale, size=self.state_dim)

    def zero_state(self) -> np.ndarray:
        """Return the zero state of the motion model."""
        return np.zeros(self.state_dim)

    def zero_action(self) -> np.ndarray:
        """Return the zero action of the motion model."""
        return np.zeros(self.action_dim)


class OmnidirectionalModel(MotionModel):
    """Omnidirectional model under `numpy`.

    Args:
        state: x, y, and theta.
        action: velocity (x and y) and angular speed.
    """
    _motion_model_type = MotionModelType.HOLONOMIC

    def __init__(self, sampling_time: float) -> None:
        super().__init__(omnidirectional_model, 3, 3, sampling_time)

class UnicycleModel(MotionModel):
    """Unicycle model under `numpy`.

    Args:
        state: x, y, and theta.
        action: speed and angular speed.
    """
    _motion_model_type = MotionModelType.NON_HOLONOMIC

    def __init__(self, sampling_time: float, rk4:bool=True) -> None:
        super().__init__(unicycle_model, 3, 2, sampling_time)
        self.rk4 = rk4

    def __call__(self, state: np.ndarray, action: np.ndarray, ts: None | float = None, **kwargs) -> np.ndarray:
        return super().__call__(state, action, ts, rk4=self.rk4)

class ReciprocatingModel(MotionModel):
    """Reciprocating preset model.

    Args:
        state: x, y, and theta.
        action: speed and angular speed.
    """
    _motion_model_type = MotionModelType.PRESET

    def __init__(self, sampling_time: float, path_nodes: list[tuple[float, float]]) -> None:
        super().__init__(reciprocating_model, 3, 1, sampling_time)
        if len(path_nodes) < 2:
            raise ValueError("At least two path nodes are required.")
        self.path_nodes = path_nodes
        self.last_node_idx = 0
        self.next_node_idx = 1

    @property
    def current_target_node(self) -> tuple[float, float]:
        """The current target node."""
        return self.path_nodes[self.next_node_idx]

    def __call__(self,  state: np.ndarray, action: np.ndarray, ts: None | float = None, **kwargs) -> np.ndarray:
        """kt: The current time step."""
        if ts is not None:
            self.ts = ts
        target_x, target_y = self.current_target_node
        if np.linalg.norm(np.array([target_x, target_y]) - state[:2]) <= action[0] * self.ts:
            if self.next_node_idx == len(self.path_nodes) - 1:
                self.last_node_idx = 0
                self.next_node_idx = 1
                self.path_nodes.reverse() # reverse the path to go back
            else:
                self.last_node_idx = self.next_node_idx
                self.next_node_idx += 1
            if isinstance(state, cs.SX):
                return cs.vertcat(target_x, target_y, state[2])
            return np.array([target_x, target_y, state[2]])
        return self.model(state, action, self.ts, target_node=self.current_target_node)


def _normalize_angle(angle: Union[float, cs.SX]) -> Union[float, cs.SX]:
    """Normalize an angle to [-pi, pi]."""
    if isinstance(angle, cs.SX):
        return cs.fmod(angle + cs.pi, 2 * cs.pi) - cs.pi
    return (angle + np.pi) % (2 * np.pi) - np.pi

def omnidirectional_model(state: Union[np.ndarray, cs.SX], action: Union[np.ndarray, cs.SX], ts: float) -> Union[np.ndarray, cs.SX]:
    """Omnidirectional model.
    
    Args:
        ts: Sampling time.
        state: x, y, and theta.
        action: velocity (x and y) and angular speed.
    """
    d_state = ts * action
    next_state = state + d_state
    next_state[2] = _normalize_angle(next_state[2])
    return next_state

def unicycle_model(state: Union[np.ndarray, cs.SX], action: Union[np.ndarray, cs.SX], ts: float, rk4:bool=True) -> Union[np.ndarray, cs.SX]:
    """Unicycle model.
    
    Args:
        ts: Sampling time.
        state: x, y, and theta.
        action: speed and angular speed.
        rk4: If True, use Runge-Kutta 4 to refine the model.
    """
    def d_state_f(state, action):
        if isinstance(state, cs.SX):
            return ts * cs.vertcat(action[0]*cs.cos(state[2]), action[0]*cs.sin(state[2]), action[1])
        return ts * np.array([action[0]*np.cos(state[2]), action[0]*np.sin(state[2]), action[1]])
    if rk4:
        k1 = d_state_f(state, action)
        k2 = d_state_f(state + 0.5*k1, action)
        k3 = d_state_f(state + 0.5*k2, action)
        k4 = d_state_f(state + k3, action)
        d_state = (1/6) * (k1 + 2*k2 + 2*k3 + k4)
    else:
        d_state = d_state_f(state, action)
    next_state = state + d_state
    next_state[2] = _normalize_angle(next_state[2])
    return next_state

def reciprocating_model(state: Union[np.ndarray, cs.SX], action: Union[np.ndarray, cs.SX], ts: float, target_node: tuple[float, float]) -> Union[np.ndarray, cs.SX]:
    """Reciprocating model moving along a pre-defined path.
    
    Args:
        state: x, y, theta.
        action: speed.
        ts: Sampling time.
        target_node: The given target node to move towards.
    """
    x, y, theta = state
    target_x, target_y = target_node
    dx = target_x - x
    dy = target_y - y
    target_theta = np.arctan2(dy, dx)
    x += action[0] * ts * np.cos(theta)
    y += action[0] * ts * np.sin(theta)
    theta = _normalize_angle(target_theta)
    if isinstance(state, cs.SX):
        return cs.vertcat(x, y, theta)
    return np.array([x, y, theta])

def car_model(state: Union[np.ndarray, cs.SX], action: Union[np.ndarray, cs.SX], ts: float) -> Union[np.ndarray, cs.SX]:
    """http://msl.cs.uiuc.edu/planning/node658.html"""
    raise NotImplementedError


if __name__ == "__main__":
    import matplotlib.pyplot as plt # type: ignore

    path_nodes = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    model = ReciprocatingModel(sampling_time=0.1, path_nodes=path_nodes)
    state = np.asarray([path_nodes[0][0], path_nodes[0][1], 0.0])

    fig, ax = plt.subplots()
    for i in range(100):
        ax.cla()
        ax.plot([p[0] for p in path_nodes], [p[1] for p in path_nodes], 'k--')
        state = model.f_noise(state, np.array([0.4]), noise_scale=0.01)
        ax.plot(state[0], state[1], 'ro')
        ax.axis('equal')
        plt.pause(0.1)

    plt.show()