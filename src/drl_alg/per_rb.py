from typing import Any, Union, NamedTuple

import numpy as np

import torch
from gymnasium import spaces

from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.vec_env import VecNormalize


class PerReplayBufferSamples(NamedTuple):
    observations: TensorDict
    actions: torch.Tensor
    next_observations: TensorDict
    dones: torch.Tensor
    rewards: torch.Tensor
    indices: np.ndarray
    weights: np.ndarray

    @property
    def discounts(self) -> None:
        """One-step PER samples use the algorithm gamma (SB3 2.7 API)."""
        return None


class PerReplayBuffer(DictReplayBuffer):
    """Replay buffer with Prioritized experience replay using a sum-tree data
    structure (https://doi.org/10.48550/arXiv.1511.05952).

    Args:
        buffer_size: Max number of element in the buffer
        observation_space: Observation space
        action_space: Action space
        device: PyTorch device
        n_envs: Number of parallel environments
        optimize_memory_usage: Enable a memory efficient variant. Disabled for now.
            (https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702)
        handle_timeout_termination: Handle timeout termination (due to timelimit)
            separately and treat the task as infinite horizon task 
            (https://github.com/DLR-RM/stable-baselines3/issues/284).
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        alpha = 0.3,
        beta = 0.4,
        epsilon = 1e-3,
        update_max_freq = 1_000,
        refresh_tree_freq = 50_000,
        initial_priority = 1,
    ):
        assert optimize_memory_usage is False, "PerReplayBuffer does not support optimize_memory_usage"

        super().__init__(buffer_size, observation_space, action_space, device, 
                         1, optimize_memory_usage, handle_timeout_termination)

        self._n_envs = n_envs
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.update_max_freq = update_max_freq
        self.refresh_tree_freq = refresh_tree_freq
        self.initial_priority = initial_priority

        self.tree = np.zeros((2 * self.buffer_size - 1,))
        self.update_max_count = self.update_max_freq - 1
        self.refresh_tree_count = 0

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def update_priority(self, idx: int, delta: float) -> None:
        self._update_priority(idx, (np.abs(delta) + self.epsilon)**self.alpha)

    def _update_priority(self, idx: int, p: float) -> None:
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def _refresh_tree(self) -> None:
        self.tree[0:self.buffer_size - 1] = 0
        for idx in range(self.buffer_size - 1, len(self.tree)):
            self._propagate(idx, self.tree[idx])
    
    def add(
        self,
        obs: dict[str, np.ndarray],
        next_obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]]
    ) -> None:
        for i in range(self._n_envs):
            self.update_max_count += 1
            if self.update_max_count >= self.update_max_freq:
                if self.pos == 0 and not self.full:
                    self.max_p = self.initial_priority
                else:
                    self.max_p = np.max(self.tree[self.buffer_size - 1:])
                self.update_max_count = 0

            idx = self.pos + self.buffer_size - 1

            super().add(
                {k: v[i:i + 1] for k, v in obs.items()},
                {k: v[i:i + 1] for k, v in next_obs.items()},
                action[i:i + 1],
                reward[i:i + 1],
                done[i:i + 1],
                infos[i:i + 1]
            )

            self._update_priority(idx, self.max_p)

            self.refresh_tree_count += 1
            if self.refresh_tree_count >= self.refresh_tree_freq:
                self._refresh_tree()
                self.refresh_tree_count = 0

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def sample(self, batch_size: int, env: VecNormalize | None = None) -> PerReplayBufferSamples:
        indices = np.zeros((batch_size,), dtype=int)

        while True:
            segment = self.tree[0] / batch_size

            for i in range(batch_size):
                a = segment * i
                b = segment * (i + 1)

                s = np.random.uniform(a, b)
                indices[i] = self._retrieve(0, s)
                if self.tree[indices[i]] == 0:
                    break
            else:
                break

            self._refresh_tree()

        data_idx = indices - self.buffer_size + 1

        n_entries = self.buffer_size if self.full else self.pos

        weights = np.power(n_entries * self.tree[indices] / self.tree[0], -self.beta)
        weights /= np.max(weights)

        sample = self._get_samples(data_idx, env)
        sample_ = (sample.observations, sample.actions, sample.next_observations, sample.dones, sample.rewards)

        return PerReplayBufferSamples(*sample_, indices, weights)

    def reset(self) -> None:
        super().reset()
        self.tree[:] = 0
        self.update_max_count = self.update_max_freq - 1
        self.refresh_tree_count = 0