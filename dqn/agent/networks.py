from __future__ import annotations
import torch
import torch.nn as nn

BOARD_ROWS = 20
BOARD_COLS = 10


class QNetwork(nn.Module):
    """
    Maps a single afterstate board (20x10 binary grid) to a scalar Q-value.
    Applied independently to each candidate placement; the agent picks the max.
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
        self.head = nn.Sequential(
            nn.Linear(64 * 10 * 5, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, boards: torch.Tensor) -> torch.Tensor:
        """boards: (B, 1, 20, 10) -> (B,)"""
        return self.head(self.cnn(boards)).squeeze(-1)
