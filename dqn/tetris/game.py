from __future__ import annotations
import copy
from typing import Dict, Optional, Tuple

import numpy as np

from tetris.constants import (
    BOARD_ROWS, BOARD_COLS, BOARD_BUFFER,
    LINE_SCORE, BACK_TO_BACK_MULTIPLIER,
    SPAWN_ROW, SPAWN_COL,
)
from tetris.pieces import Piece, SevenBag, PIECE_IDS

QUEUE_LEN = 5
FULL_ROWS = BOARD_ROWS + BOARD_BUFFER  # includes hidden buffer


class Game:
    def __init__(self, seed: int = None):
        self.seed = seed
        self._bag = SevenBag(seed)
        self.reset()

    def reset(self):
        self._bag = SevenBag(self.seed)
        # Board includes buffer rows at the top
        self.board = np.zeros((FULL_ROWS, BOARD_COLS), dtype=np.int8)
        self.hold: Optional[str] = None
        self.hold_used = False
        self.queue = [self._bag.next() for _ in range(QUEUE_LEN)]
        self.score = 0
        self.lines_cleared = 0
        self.pieces_placed = 0
        self.game_over = False
        self.back_to_back = False  # tracks b2b tetris streak
        self.active_piece = self._spawn()

    # ------------------------------------------------------------------
    # Piece spawning
    # ------------------------------------------------------------------

    def _spawn(self, piece_type: str = None) -> Optional[Piece]:
        if piece_type is None:
            piece_type = self.queue.pop(0)
            self.queue.append(self._bag.next())
        piece = Piece(piece_type, rotation=0, row=BOARD_BUFFER - 1, col=SPAWN_COL)
        if self._collides(piece):
            return None  # spawn blocked → game over
        return piece

    # ------------------------------------------------------------------
    # Collision and movement
    # ------------------------------------------------------------------

    def _collides(self, piece: Piece) -> bool:
        for r, c in piece.cells():
            if c < 0 or c >= BOARD_COLS:
                return True
            if r >= FULL_ROWS:
                return True
            if r >= 0 and self.board[r, c] != 0:
                return True
        return False

    def _lock(self, piece: Piece):
        for r, c in piece.cells():
            if 0 <= r < FULL_ROWS:
                self.board[r, c] = piece.piece_id

    def _drop_to_floor(self, piece: Piece) -> Piece:
        """Hard-drop: move piece down until it would collide."""
        while not self._collides(piece.translate(1, 0)):
            piece = piece.translate(1, 0)
        return piece

    # ------------------------------------------------------------------
    # High-level placement action (used by RL env)
    # ------------------------------------------------------------------

    def place_piece(self, rotation: int, col: int, use_hold: bool = False) -> Dict:
        """
        Place the current (or held) piece at a given column + rotation.
        Returns info dict with lines cleared and reward components.
        Returns None action info if the placement is illegal.
        """
        if self.game_over:
            return {"valid": False, "lines": 0, "reward": 0.0}

        if use_hold:
            success = self._do_hold()
            if not success:
                return {"valid": False, "lines": 0, "reward": 0.0}

        piece = Piece(self.active_piece.piece_type, rotation % 4, BOARD_BUFFER - 1, col)
        if self._collides(piece):
            return {"valid": False, "lines": 0, "reward": 0.0}

        piece = self._drop_to_floor(piece)
        self._lock(piece)
        self.pieces_placed += 1
        self.hold_used = False

        n_cleared = self._clear_lines()
        self.lines_cleared += n_cleared

        # Scoring
        is_tetris = n_cleared == 4
        if n_cleared > 0:
            points = LINE_SCORE[n_cleared]
            if is_tetris and self.back_to_back:
                points = int(points * BACK_TO_BACK_MULTIPLIER)
            self.score += points
        self.back_to_back = is_tetris

        # Spawn next piece
        next_piece = self._spawn()
        if next_piece is None:
            self.game_over = True
            self.active_piece = Piece(self.queue[0])  # placeholder
            return {"valid": True, "lines": n_cleared, "game_over": True}

        self.active_piece = next_piece
        return {"valid": True, "lines": n_cleared, "game_over": False}

    def _do_hold(self) -> bool:
        if self.hold_used:
            return False
        if self.hold is None:
            self.hold = self.active_piece.piece_type
            next_piece = self._spawn()
            if next_piece is None:
                self.game_over = True
                return False
            self.active_piece = next_piece
        else:
            prev_hold = self.hold
            self.hold = self.active_piece.piece_type
            piece = Piece(prev_hold, 0, BOARD_BUFFER - 1, SPAWN_COL)
            if self._collides(piece):
                self.hold = self.active_piece.piece_type  # revert
                return False
            self.active_piece = piece
        self.hold_used = True
        return True

    # ------------------------------------------------------------------
    # Line clearing
    # ------------------------------------------------------------------

    def _clear_lines(self) -> int:
        full = np.all(self.board != 0, axis=1)
        n = int(full.sum())
        if n == 0:
            return 0
        remaining = self.board[~full]
        empty = np.zeros((n, BOARD_COLS), dtype=np.int8)
        self.board = np.vstack([empty, remaining])
        return n

    # ------------------------------------------------------------------
    # Action masking
    # ------------------------------------------------------------------

    def get_action_mask(self) -> np.ndarray:
        """
        Returns bool array of length 81.
        Actions 0-39:  place current piece (rot*10 + col)
        Actions 40-79: place held piece after swap
        Action 80:     hold current piece (only if hold unused)
        """
        mask = np.zeros(81, dtype=bool)

        def _check_placement(piece_type: str):
            valid = np.zeros(40, dtype=bool)
            for rot in range(4):
                for col in range(BOARD_COLS):
                    p = Piece(piece_type, rot, BOARD_BUFFER - 1, col)
                    if not self._collides(p):
                        valid[rot * BOARD_COLS + col] = True
            return valid

        # Current piece placements
        mask[0:40] = _check_placement(self.active_piece.piece_type)

        # Hold piece placements
        if not self.hold_used:
            hold_type = self.hold if self.hold else self.queue[0]
            if self.hold is None:
                # Holding would consume next from queue; use queue[0] to check
                hold_type = self.queue[0]
            mask[40:80] = _check_placement(hold_type)

        # Hold-only action: valid if hold slot unused and not game over
        if not self.hold_used:
            mask[80] = True

        return mask

    # ------------------------------------------------------------------
    # Observable state
    # ------------------------------------------------------------------

    def get_state(self) -> Dict:
        return {
            "board": self.board[BOARD_BUFFER:].copy(),  # visible rows only
            "current_piece": PIECE_IDS[self.active_piece.piece_type] - 1,
            "hold": (PIECE_IDS[self.hold] - 1) if self.hold else 7,
            "queue": [PIECE_IDS[p] - 1 for p in self.queue],
            "score": self.score,
            "lines_cleared": self.lines_cleared,
            "game_over": self.game_over,
            "action_mask": self.get_action_mask(),
        }

    # ------------------------------------------------------------------
    # Clone for search
    # ------------------------------------------------------------------

    def clone(self) -> "Game":
        g = Game.__new__(Game)
        g.seed = self.seed
        g._bag = self._bag.clone()
        g.board = self.board.copy()
        g.hold = self.hold
        g.hold_used = self.hold_used
        g.queue = list(self.queue)
        g.score = self.score
        g.lines_cleared = self.lines_cleared
        g.pieces_placed = self.pieces_placed
        g.game_over = self.game_over
        g.back_to_back = self.back_to_back
        g.active_piece = self.active_piece.clone()
        return g
