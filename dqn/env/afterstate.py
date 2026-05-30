from __future__ import annotations
import numpy as np
from tetris.constants import BOARD_ROWS, BOARD_COLS, BOARD_BUFFER
from tetris.pieces import PIECE_IDS

MAX_ACTIONS = 40   # 4 rotations x 10 columns for current piece
QUEUE_LEN   = 3    # how many upcoming pieces to feed to the Q-network


def compute_afterstates(game) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate all valid placements of the current piece.

    Returns
    -------
    boards     : (MAX_ACTIONS, BOARD_ROWS, BOARD_COLS) float32
        Binary board after placement. Zero for invalid actions.
    lines      : (MAX_ACTIONS,) int32
        Lines cleared. Zero for invalid actions.
    mask       : (MAX_ACTIONS,) bool
        True where the action is legal.
    next_queue : (QUEUE_LEN,) int64
        Indices (0-6) of the next QUEUE_LEN pieces in the bag.
        Identical for every placement of the same current piece.
    """
    action_mask = game.get_action_mask()[:MAX_ACTIONS]
    boards = np.zeros((MAX_ACTIONS, BOARD_ROWS, BOARD_COLS), dtype=np.float32)
    lines  = np.zeros(MAX_ACTIONS, dtype=np.int32)
    mask   = np.zeros(MAX_ACTIONS, dtype=bool)

    for idx in np.where(action_mask)[0]:
        rot = int(idx) // BOARD_COLS
        col = int(idx) % BOARD_COLS
        g = game.clone()
        info = g.place_piece(rot, col, use_hold=False)
        if info.get("valid", False):
            mask[idx] = True
            boards[idx] = (g.board[BOARD_BUFFER:] > 0).astype(np.float32)
            lines[idx]  = int(info.get("lines", 0))

    # The next queue is the same regardless of which placement is chosen
    next_queue = np.array(
        [PIECE_IDS[p] - 1 for p in game.queue[:QUEUE_LEN]], dtype=np.int64
    )
    return boards, lines, mask, next_queue
