BOARD_ROWS = 20
BOARD_COLS = 10
BOARD_BUFFER = 2  # hidden rows above for spawn

PIECE_TYPES = ["I", "O", "T", "S", "Z", "J", "L"]
NUM_PIECES = 7

# Scoring: lines cleared -> points
LINE_SCORE = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
BACK_TO_BACK_MULTIPLIER = 1.5

LOCK_DELAY_TICKS = 30  # ~500ms at 60fps
MAX_LOCK_RESETS = 15

SPAWN_ROW = BOARD_BUFFER - 1  # row index where pieces spawn (in full grid)
SPAWN_COL = BOARD_COLS // 2 - 1

COLORS = {
    0: (0, 0, 0),
    1: (0, 240, 240),   # I - cyan
    2: (240, 240, 0),   # O - yellow
    3: (160, 0, 240),   # T - purple
    4: (0, 240, 0),     # S - green
    5: (240, 0, 0),     # Z - red
    6: (0, 0, 240),     # J - blue
    7: (240, 160, 0),   # L - orange
    8: (128, 128, 128), # ghost
}

CELL_SIZE = 30
BOARD_OFFSET_X = 200
BOARD_OFFSET_Y = 40
PANEL_X = BOARD_OFFSET_X + BOARD_COLS * CELL_SIZE + 20
WINDOW_WIDTH = 700
WINDOW_HEIGHT = BOARD_ROWS * CELL_SIZE + 2 * BOARD_OFFSET_Y
