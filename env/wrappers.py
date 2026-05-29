from __future__ import annotations
from typing import Any, Dict, List, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class RewardClipWrapper(gym.RewardWrapper):
    """Clip rewards to [-max_abs, max_abs]."""

    def __init__(self, env: gym.Env, max_abs: float = 10.0):
        super().__init__(env)
        self._max_abs = max_abs

    def reward(self, reward: float) -> float:
        return float(np.clip(reward, -self._max_abs, self._max_abs))


class RecordEpisodeStatistics(gym.Wrapper):
    """Accumulates per-episode stats and adds them to the final step's info."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._ep_return = 0.0
        self._ep_length = 0
        self._ep_lines = 0

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._ep_return = 0.0
        self._ep_length = 0
        self._ep_lines = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._ep_return += reward
        self._ep_length += 1
        self._ep_lines = info.get("lines_cleared", self._ep_lines)
        if terminated or truncated:
            info["episode"] = {
                "r": self._ep_return,
                "l": self._ep_length,
                "lines": self._ep_lines,
                "score": info.get("score", 0),
                "pieces": info.get("pieces_placed", 0),
            }
        return obs, reward, terminated, truncated, info


class ObsNormWrapper(gym.ObservationWrapper):
    """
    Flattens the Dict observation into a single float32 vector.
    board (200) + current_piece (7 one-hot) + hold (8 one-hot) + queue (5*7 one-hot) = 265
    """

    PIECE_DIM = 7
    HOLD_DIM = 8
    QUEUE_DIM = 7
    QUEUE_LEN = 5

    def __init__(self, env: gym.Env):
        super().__init__(env)
        flat_dim = 200 + self.PIECE_DIM + self.HOLD_DIM + self.QUEUE_LEN * self.QUEUE_DIM
        self.observation_space = spaces.Dict({
            **{k: v for k, v in env.observation_space.items() if k != "board"},
            "board": spaces.Box(0.0, 1.0, shape=(200,), dtype=np.float32),
        })
        # Actually expose full flat vector
        self.observation_space = spaces.Box(0.0, 1.0, shape=(flat_dim,), dtype=np.float32)
        self._action_mask_space = env.observation_space["action_mask"]

    def observation(self, obs: Dict) -> np.ndarray:
        board_flat = obs["board"].flatten()

        cp = np.zeros(self.PIECE_DIM, dtype=np.float32)
        cp[obs["current_piece"]] = 1.0

        hold = np.zeros(self.HOLD_DIM, dtype=np.float32)
        hold[obs["hold"]] = 1.0

        queue = np.zeros(self.QUEUE_LEN * self.QUEUE_DIM, dtype=np.float32)
        for i, p in enumerate(obs["queue"]):
            queue[i * self.QUEUE_DIM + p] = 1.0

        return np.concatenate([board_flat, cp, hold, queue]).astype(np.float32)

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()


def make_env(seed: int = None, record_stats: bool = True) -> gym.Env:
    from env.tetris_env import TetrisEnv
    env = TetrisEnv(seed=seed)
    if record_stats:
        env = RecordEpisodeStatistics(env)
    return env
