from __future__ import annotations
import torch
import torch.nn as nn
from env.afterstate import QUEUE_LEN

BOARD_ROWS = 20
BOARD_COLS = 10
N_PIECES   = 7


class QNetwork(nn.Module):
    """
    Maps (afterstate board, upcoming piece queue) -> scalar Q-value.

    Knowing what pieces are coming is critical for Tetris planning —
    the same board position has very different value depending on whether
    an I-piece or an S-piece is next.
    """

    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (B, 32, 20, 10)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B, 64, 20, 10)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # (B, 64, 10,  5)
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # (B, 64, 10,  5)
            nn.ReLU(),
            nn.Flatten(),                                  # (B, 3200)
        )
        cnn_out = 64 * 10 * 5  # 3200

        # Embed each queued piece as a 16-dim vector then flatten
        self.piece_embed = nn.Embedding(N_PIECES, 16)
        queue_dim = QUEUE_LEN * 16  # 48

        self.head = nn.Sequential(
            nn.Linear(cnn_out + queue_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, boards: torch.Tensor, queue: torch.Tensor) -> torch.Tensor:
        """
        boards : (B, 1, 20, 10) float32
        queue  : (B, QUEUE_LEN) int64
        returns: (B,) float32
        """
        board_feat = self.cnn(boards)                    # (B, 3200)
        queue_feat = self.piece_embed(queue).flatten(1)  # (B, 48)
        return self.head(torch.cat([board_feat, queue_feat], dim=1)).squeeze(-1)
