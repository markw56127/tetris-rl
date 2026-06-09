from __future__ import annotations

import gymnasium as gym


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


def make_env(seed: int = None, record_stats: bool = True) -> gym.Env:
    from env.tetris_env import TetrisEnv
    env = TetrisEnv(seed=seed)
    if record_stats:
        env = RecordEpisodeStatistics(env)
    return env
