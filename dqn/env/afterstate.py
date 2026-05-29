from __future__ import annotations
import numpy as np
from tetris.constants import BOARD_ROWS, BOARD_COLS, BOARD_BUFFER

MAX_ACTIONS = 40  # 4 rotations x 10 columns for current piece


def compute_afterstates(game) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each of the 40 possible (rotation, col) placements of the current piece,
    simulate the result and return the afterstate board.

    Returns
    -------
    boards : (MAX_ACTIONS, BOARD_ROWS, BOARD_COLS) float32
        Binary board after placement. Zero for invalid actions.
    lines  : (MAX_ACTIONS,) int32
        Lines cleared by each placement. Zero for invalid actions.
    mask   : (MAX_ACTIONS,) bool
        True where the action is legal.
    """
    action_mask = game.get_action_mask()[:MAX_ACTIONS]
    boards = np.zeros((MAX_ACTIONS, BOARD_ROWS, BOARD_COLS), dtype=np.float32)
    lines = np.zeros(MAX_ACTIONS, dtype=np.int32)
    mask = np.zeros(MAX_ACTIONS, dtype=bool)

    for idx in np.where(action_mask)[0]:
        rot = int(idx) // BOARD_COLS
        col = int(idx) % BOARD_COLS
        g = game.clone()
        info = g.place_piece(rot, col, use_hold=False)
        if info.get("valid", False):
            mask[idx] = True
            boards[idx] = (g.board[BOARD_BUFFER:] > 0).astype(np.float32)
            lines[idx] = int(info.get("lines", 0))

    return boards, lines, mask
