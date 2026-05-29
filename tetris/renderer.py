"""Optional pygame renderer. Only imported for human play / evaluation."""
from typing import Optional

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from tetris.constants import (
    BOARD_ROWS, BOARD_COLS, BOARD_BUFFER, COLORS, CELL_SIZE,
    BOARD_OFFSET_X, BOARD_OFFSET_Y, PANEL_X, WINDOW_WIDTH, WINDOW_HEIGHT,
)
from tetris.pieces import PIECE_IDS


class Renderer:
    def __init__(self):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is required for rendering. Run: pip install pygame")
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Tetris RL")
        self.font = pygame.font.SysFont("monospace", 20)
        self.clock = pygame.time.Clock()

    def draw(self, game) -> bool:
        """Draw current game state. Returns False if user closed the window."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        self.screen.fill((20, 20, 20))
        self._draw_board(game)
        self._draw_active_piece(game)
        self._draw_ghost(game)
        self._draw_panel(game)
        pygame.display.flip()
        self.clock.tick(60)
        return True

    def _draw_board(self, game):
        board = game.board[BOARD_BUFFER:]
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                cell = board[r, c]
                x = BOARD_OFFSET_X + c * CELL_SIZE
                y = BOARD_OFFSET_Y + r * CELL_SIZE
                color = COLORS.get(cell, COLORS[0])
                pygame.draw.rect(self.screen, color, (x, y, CELL_SIZE - 1, CELL_SIZE - 1))

        # Border
        pygame.draw.rect(
            self.screen, (80, 80, 80),
            (BOARD_OFFSET_X - 1, BOARD_OFFSET_Y - 1,
             BOARD_COLS * CELL_SIZE + 2, BOARD_ROWS * CELL_SIZE + 2), 1
        )

    def _draw_active_piece(self, game):
        piece = game.active_piece
        for r, c in piece.cells():
            vr = r - BOARD_BUFFER
            if vr < 0:
                continue
            x = BOARD_OFFSET_X + c * CELL_SIZE
            y = BOARD_OFFSET_Y + vr * CELL_SIZE
            pygame.draw.rect(self.screen, COLORS[piece.piece_id], (x, y, CELL_SIZE - 1, CELL_SIZE - 1))

    def _draw_ghost(self, game):
        piece = game.active_piece
        ghost = piece
        while not game._collides(ghost.translate(1, 0)):
            ghost = ghost.translate(1, 0)
        for r, c in ghost.cells():
            vr = r - BOARD_BUFFER
            if vr < 0:
                continue
            x = BOARD_OFFSET_X + c * CELL_SIZE
            y = BOARD_OFFSET_Y + vr * CELL_SIZE
            pygame.draw.rect(self.screen, COLORS[8], (x, y, CELL_SIZE - 1, CELL_SIZE - 1))

    def _draw_panel(self, game):
        state = game.get_state()

        def label(text, y):
            surf = self.font.render(text, True, (200, 200, 200))
            self.screen.blit(surf, (PANEL_X, y))

        label(f"Score:  {state['score']}", 40)
        label(f"Lines:  {state['lines_cleared']}", 70)
        label(f"Pieces: {game.pieces_placed}", 100)

        label("HOLD", 150)
        if game.hold:
            self._draw_mini_piece(game.hold, PANEL_X, 175)

        label("NEXT", 260)
        for i, ptype in enumerate(game.queue[:5]):
            self._draw_mini_piece(ptype, PANEL_X, 285 + i * 75)

    def _draw_mini_piece(self, piece_type: str, x: int, y: int):
        from tetris.pieces import Piece, PIECE_ROTATIONS
        offsets = PIECE_ROTATIONS[piece_type][0]
        pid = PIECE_IDS[piece_type]
        min_r = min(dr for dr, dc in offsets)
        min_c = min(dc for dr, dc in offsets)
        s = 18
        for dr, dc in offsets:
            px = x + (dc - min_c) * s
            py = y + (dr - min_r) * s
            pygame.draw.rect(self.screen, COLORS[pid], (px, py, s - 1, s - 1))

    def close(self):
        pygame.quit()
