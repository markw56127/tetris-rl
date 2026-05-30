from __future__ import annotations
import random
import numpy as np
from collections import deque

from env.afterstate import MAX_ACTIONS, QUEUE_LEN


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        afterstate:  np.ndarray,  # (H, W)         float32 — chosen board
        queue:       np.ndarray,  # (QUEUE_LEN,)    int64   — queue at this step
        reward:      float,
        next_boards: np.ndarray,  # (MAX_ACTIONS, H, W) float32
        next_queue:  np.ndarray,  # (QUEUE_LEN,)    int64   — queue at next step
        next_mask:   np.ndarray,  # (MAX_ACTIONS,)  float32
        done:        bool,
    ) -> None:
        self.buffer.append((
            afterstate.astype(np.float32),
            queue.astype(np.int64),
            float(reward),
            next_boards.astype(np.float32),
            next_queue.astype(np.int64),
            next_mask.astype(np.float32),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        afterstates, queues, rewards, next_boards, next_queues, next_masks, dones = zip(*batch)
        return (
            np.array(afterstates),                   # (B, H, W)
            np.array(queues,      dtype=np.int64),   # (B, QUEUE_LEN)
            np.array(rewards,     dtype=np.float32), # (B,)
            np.array(next_boards),                   # (B, 40, H, W)
            np.array(next_queues, dtype=np.int64),   # (B, QUEUE_LEN)
            np.array(next_masks,  dtype=np.float32), # (B, 40)
            np.array(dones,       dtype=np.float32), # (B,)
        )

    def __len__(self) -> int:
        return len(self.buffer)
