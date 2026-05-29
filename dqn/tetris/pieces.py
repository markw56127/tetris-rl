import random
from typing import List, Tuple

# SRS rotation states: each piece is a list of 4 rotations,
# each rotation is a list of (row, col) offsets from pivot.
PIECE_ROTATIONS = {
    "I": [
        [(0, -1), (0, 0), (0, 1), (0, 2)],
        [(-1, 1), (0, 1), (1, 1), (2, 1)],
        [(1, -1), (1, 0), (1, 1), (1, 2)],
        [(-1, 0), (0, 0), (1, 0), (2, 0)],
    ],
    "O": [
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ],
    "T": [
        [(0, -1), (0, 0), (0, 1), (1, 0)],
        [(-1, 0), (0, 0), (0, 1), (1, 0)],
        [(-1, 0), (0, -1), (0, 0), (0, 1)],
        [(-1, 0), (0, -1), (0, 0), (1, 0)],
    ],
    "S": [
        [(0, 0), (0, 1), (1, -1), (1, 0)],
        [(-1, 0), (0, 0), (0, 1), (1, 1)],
        [(0, 0), (0, 1), (1, -1), (1, 0)],
        [(-1, 0), (0, 0), (0, 1), (1, 1)],
    ],
    "Z": [
        [(0, -1), (0, 0), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],  # corrected for visual correctness
        [(0, -1), (0, 0), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
    ],
    "J": [
        [(0, -1), (0, 0), (0, 1), (1, 1)],
        [(-1, 0), (0, 0), (1, 0), (1, -1)],
        [(-1, -1), (0, -1), (0, 0), (0, 1)],
        [(-1, 0), (-1, 1), (0, 0), (1, 0)],
    ],
    "L": [
        [(0, -1), (0, 0), (0, 1), (1, -1)],
        [(-1, 0), (0, 0), (1, 0), (1, 1)],
        [(-1, 1), (0, -1), (0, 0), (0, 1)],
        [(-1, -1), (-1, 0), (0, 0), (1, 0)],
    ],
}

PIECE_IDS = {"I": 1, "O": 2, "T": 3, "S": 4, "Z": 5, "J": 6, "L": 7}
PIECE_NAMES = {v: k for k, v in PIECE_IDS.items()}


class Piece:
    def __init__(self, piece_type: str, rotation: int = 0, row: int = 0, col: int = 0):
        self.piece_type = piece_type
        self.rotation = rotation % 4
        self.row = row
        self.col = col
        self.piece_id = PIECE_IDS[piece_type]

    def cells(self) -> List[Tuple[int, int]]:
        offsets = PIECE_ROTATIONS[self.piece_type][self.rotation]
        return [(self.row + dr, self.col + dc) for dr, dc in offsets]

    def rotate(self, direction: int = 1) -> "Piece":
        return Piece(self.piece_type, (self.rotation + direction) % 4, self.row, self.col)

    def translate(self, dr: int, dc: int) -> "Piece":
        return Piece(self.piece_type, self.rotation, self.row + dr, self.col + dc)

    def clone(self) -> "Piece":
        return Piece(self.piece_type, self.rotation, self.row, self.col)

    def __repr__(self) -> str:
        return f"Piece({self.piece_type}, rot={self.rotation}, pos=({self.row},{self.col}))"


class SevenBag:
    """7-bag randomizer: shuffle all 7 piece types, deal in order, refill when empty."""

    def __init__(self, seed: int = None):
        self._rng = random.Random(seed)
        self._bag: List[str] = []

    def _refill(self):
        self._bag = list(PIECE_IDS.keys())
        self._rng.shuffle(self._bag)

    def next(self) -> str:
        if not self._bag:
            self._refill()
        return self._bag.pop()

    def peek(self, n: int) -> List[str]:
        """Return next n pieces without consuming them, refilling as needed."""
        result = []
        extra_bags = []
        temp = list(self._bag)
        while len(result) < n:
            if not temp:
                new_bag = list(PIECE_IDS.keys())
                self._rng.shuffle(new_bag)
                extra_bags.append(new_bag)
                temp = list(new_bag)
            result.append(temp.pop())
        return result

    def clone(self) -> "SevenBag":
        cloned = SevenBag()
        cloned._rng = random.Random()
        cloned._rng.setstate(self._rng.getstate())
        cloned._bag = list(self._bag)
        return cloned
