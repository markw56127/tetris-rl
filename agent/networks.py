from __future__ import annotations
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

BOARD_ROWS = 20
BOARD_COLS = 10
NUM_PIECES = 7
HOLD_DIM = 8
QUEUE_LEN = 5


class TetrisCNNExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for Dict observation space.
    Processes the board with a small CNN, embeds piece/hold/queue,
    then concatenates everything.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # (B,32,20,10) → (B,32,10,5)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # (B,64,10,5)  → (B,64,5,2)
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        cnn_out = 64 * 5 * 2  # 640, down from 12800

        self.board_linear = nn.Sequential(
            nn.Linear(cnn_out, 256),
            nn.ReLU(),
        )

        # SB3 one-hot encodes Discrete/MultiDiscrete before passing to extractor,
        # so use Linear projections that operate directly on the one-hot vectors.
        self.piece_proj = nn.Linear(NUM_PIECES, 16)       # (B, 7)  → (B, 16)
        self.hold_proj = nn.Linear(HOLD_DIM, 16)          # (B, 8)  → (B, 16)
        self.queue_proj = nn.Linear(QUEUE_LEN * NUM_PIECES, 80)  # (B, 35) → (B, 80)

        concat_dim = 256 + 16 + 16 + 80  # 368
        self.fc = nn.Sequential(
            nn.Linear(concat_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        board = observations["board"]  # (B, 20, 10)
        board = board.unsqueeze(1)     # (B, 1, 20, 10)

        board_feat = self.board_linear(self.cnn(board))   # (B, 256)

        # Discrete: VecEnv gives (B,7) one-hot, but rollout buffer loads as (B,1,7).
        # flatten(1) handles both: (B,7)→(B,7) and (B,1,7)→(B,7).
        cp_feat   = self.piece_proj(observations["current_piece"].float().flatten(1))  # (B, 16)
        hold_feat = self.hold_proj(observations["hold"].float().flatten(1))            # (B, 16)

        q = observations["queue"]
        if q.shape[-1] == QUEUE_LEN:
            # SB3 left as integer indices (B, 5) — one-hot encode explicitly
            q = F.one_hot(q.long(), num_classes=NUM_PIECES).float()       # (B, 5, 7)
        q_feat = self.queue_proj(q.float().flatten(1))                    # (B, 80)

        combined = torch.cat([board_feat, cp_feat, hold_feat, q_feat], dim=1)  # (B, 368)
        return self.fc(combined)  # (B, 256)
