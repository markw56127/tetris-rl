from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from tetris.game import Game
from tetris.constants import BOARD_ROWS, BOARD_COLS, BOARD_BUFFER

NUM_PIECES = 7
QUEUE_LEN = 5
# Actions: 4 rot × 10 col for current piece (0-39),
#          same for held piece after swap (40-79),
#          hold-only (80)
NUM_ACTIONS = 81


class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, seed: int = None, render_mode: str = None):
        super().__init__()
        self._seed = seed
        self.render_mode = render_mode
        self._renderer = None

        self.observation_space = spaces.Dict({
            "board": spaces.Box(0, 1, shape=(BOARD_ROWS, BOARD_COLS), dtype=np.float32),
            "current_piece": spaces.Discrete(NUM_PIECES),
            "hold": spaces.Discrete(NUM_PIECES + 1),  # 7 pieces + empty(7)
            "queue": spaces.MultiDiscrete([NUM_PIECES] * QUEUE_LEN),
            "action_mask": spaces.Box(0, 1, shape=(NUM_ACTIONS,), dtype=bool),
        })
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self.game = Game(seed=seed)
        self._prev_lines = 0
        self._prev_stats = (0, 0, 0)

    def reset(self, *, seed: int = None, options: Dict = None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        self.game = Game(seed=self._seed)
        self._prev_lines = 0
        self._prev_stats = (0, 0, 0)  # agg_height, holes, bumpiness
        obs = self._get_obs()
        return obs, {}

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        assert not self.game.game_over, "step() called on finished episode"

        # Guard against unmasked evaluation: invalid actions terminate immediately.
        if not self.game.get_action_mask()[action]:
            obs = self._get_obs()
            return obs, -10.0, True, False, {
                "lines_cleared": self.game.lines_cleared,
                "score": self.game.score,
                "pieces_placed": self.game.pieces_placed,
            }

        use_hold = 40 <= action < 80
        hold_only = action == 80

        if hold_only:
            info = self._do_hold_only()
        else:
            act_idx = action % 40
            rotation = act_idx // BOARD_COLS
            col = act_idx % BOARD_COLS
            info = self.game.place_piece(rotation, col, use_hold=use_hold)

        lines = info.get("lines", 0)
        game_over = info.get("game_over", False) or self.game.game_over

        curr_stats = self._board_stats()
        reward = self._compute_reward(lines, game_over, self._prev_stats, curr_stats)
        self._prev_stats = curr_stats

        obs = self._get_obs()
        terminated = game_over
        truncated = False
        info["score"] = self.game.score
        info["lines_cleared"] = self.game.lines_cleared
        info["pieces_placed"] = self.game.pieces_placed
        return obs, reward, terminated, truncated, info

    def _do_hold_only(self) -> Dict:
        if self.game.hold_used:
            return {"valid": False, "lines": 0, "game_over": False}
        prev_hold = self.game.hold
        self.game._do_hold()
        return {"valid": True, "lines": 0, "game_over": self.game.game_over}

    def _board_stats(self):
        board = self.game.board[BOARD_BUFFER:]  # (20, 10) visible rows
        heights = np.zeros(BOARD_COLS, dtype=np.int32)
        holes = 0
        for c in range(BOARD_COLS):
            col = board[:, c]
            nz = np.nonzero(col)[0]
            if len(nz) > 0:
                top = nz[0]
                heights[c] = BOARD_ROWS - top
                holes += int((col[top:] == 0).sum())
        agg_height = int(heights.sum())
        bumpiness = int(np.abs(np.diff(heights)).sum())
        return agg_height, holes, bumpiness

    def _compute_reward(self, lines: int, game_over: bool,
                        prev_stats: tuple, curr_stats: tuple) -> float:
        prev_agg, prev_holes, prev_bump = prev_stats
        curr_agg, curr_holes, curr_bump = curr_stats

        # Survival bonus: being alive is always better than dying
        reward = 0.5

        # Line clears (scaled up so they're meaningful relative to shaping)
        reward += float(lines ** 2) * 5

        # Board quality deltas: guide behavior but don't dominate survival bonus
        reward -= 0.1  * (curr_holes - prev_holes)
        reward -= 0.01 * (curr_agg   - prev_agg)
        reward -= 0.01 * (curr_bump  - prev_bump)

        if game_over:
            reward -= 10.0
        return reward

    def _get_obs(self) -> Dict:
        state = self.game.get_state()
        return {
            "board": (state["board"] > 0).astype(np.float32),
            "current_piece": int(state["current_piece"]),
            "hold": int(state["hold"]),
            "queue": np.array(state["queue"], dtype=np.int64),
            "action_mask": state["action_mask"],
        }

    def action_masks(self) -> np.ndarray:
        """sb3-contrib MaskablePPO interface."""
        return self.game.get_action_mask()

    def render(self):
        if self.render_mode == "human":
            if self._renderer is None:
                from tetris.renderer import Renderer
                self._renderer = Renderer()
            return self._renderer.draw(self.game)
        elif self.render_mode == "rgb_array":
            return self._render_rgb()
        return None

    def _render_rgb(self) -> np.ndarray:
        from tetris.constants import COLORS, BOARD_BUFFER
        board = self.game.board[BOARD_BUFFER:]
        img = np.zeros((BOARD_ROWS, BOARD_COLS, 3), dtype=np.uint8)
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                img[r, c] = COLORS.get(int(board[r, c]), (0, 0, 0))[:3]
        return img

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
