from __future__ import annotations
from typing import Generator, NamedTuple, Optional

import numpy as np
import torch


class RolloutBatch(NamedTuple):
    obs: dict
    actions: torch.Tensor
    log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor
    action_masks: torch.Tensor


class RolloutBuffer:
    """
    Stores n_steps × n_envs transitions, computes GAE advantages,
    and yields minibatches for PPO updates.
    """

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_keys: list,
        obs_shapes: dict,
        action_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
    ):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.obs_keys = obs_keys
        self.obs_shapes = obs_shapes
        self.action_dim = action_dim
        self._reset()

    def _reset(self):
        T, N = self.n_steps, self.n_envs
        self.obs = {k: np.zeros((T, N, *self.obs_shapes[k]), dtype=np.float32)
                    for k in self.obs_keys}
        self.actions = np.zeros((T, N), dtype=np.int64)
        self.log_probs = np.zeros((T, N), dtype=np.float32)
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.values = np.zeros((T, N), dtype=np.float32)
        self.dones = np.zeros((T, N), dtype=np.float32)
        self.action_masks = np.zeros((T, N, self.action_dim), dtype=bool)
        self.pos = 0
        self.full = False

    def add(self, obs: dict, actions, log_probs, rewards, values, dones, action_masks):
        for k in self.obs_keys:
            self.obs[k][self.pos] = obs[k]
        self.actions[self.pos] = actions
        self.log_probs[self.pos] = log_probs
        self.rewards[self.pos] = rewards
        self.values[self.pos] = values
        self.dones[self.pos] = dones
        self.action_masks[self.pos] = action_masks
        self.pos += 1
        if self.pos == self.n_steps:
            self.full = True

    def compute_returns_and_advantages(self, last_values: np.ndarray, last_dones: np.ndarray):
        """GAE-λ advantage estimation."""
        advantages = np.zeros_like(self.rewards)
        last_gae = 0.0

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]

            delta = (self.rewards[t]
                     + self.gamma * next_values * next_non_terminal
                     - self.values[t])
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        self.returns = advantages + self.values
        self.advantages = advantages

    def get_minibatches(self, batch_size: int) -> Generator[RolloutBatch, None, None]:
        T, N = self.n_steps, self.n_envs
        total = T * N
        indices = np.random.permutation(total)

        # Flatten time × envs
        flat_obs = {k: self.obs[k].reshape(total, *self.obs_shapes[k]) for k in self.obs_keys}
        flat_actions = self.actions.reshape(total)
        flat_log_probs = self.log_probs.reshape(total)
        flat_returns = self.returns.reshape(total)
        flat_advantages = self.advantages.reshape(total)
        flat_values = self.values.reshape(total)
        flat_masks = self.action_masks.reshape(total, self.action_dim)

        for start in range(0, total, batch_size):
            idx = indices[start:start + batch_size]
            yield RolloutBatch(
                obs={k: torch.tensor(flat_obs[k][idx], device=self.device) for k in self.obs_keys},
                actions=torch.tensor(flat_actions[idx], device=self.device),
                log_probs=torch.tensor(flat_log_probs[idx], device=self.device),
                returns=torch.tensor(flat_returns[idx], device=self.device),
                advantages=torch.tensor(flat_advantages[idx], device=self.device),
                values=torch.tensor(flat_values[idx], device=self.device),
                action_masks=torch.tensor(flat_masks[idx], device=self.device),
            )
