#!/usr/bin/env python3
"""
Afterstate DQN for Tetris.

Key idea: instead of learning Q(board, action), learn Q(afterstate) where
afterstate is the board after a piece has been placed and lines cleared.
The agent enumerates all valid placements, scores each resulting board with
the Q-network, and picks the best one. Double DQN reduces overestimation.
"""
import os
import copy
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from tetris.game import Game
from env.afterstate import compute_afterstates, MAX_ACTIONS
from agent.networks import QNetwork
from agent.replay_buffer import ReplayBuffer

LINE_REWARDS = [0.0, 1.0, 3.0, 5.0, 8.0]


def reward_fn(lines: int, game_over: bool) -> float:
    survival = 0.0 if game_over else 0.1
    return LINE_REWARDS[min(lines, 4)] + survival - (1.0 if game_over else 0.0)


def get_device(name: str) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def select_action(
    q_net: QNetwork,
    boards: np.ndarray,
    mask: np.ndarray,
    epsilon: float,
    device: torch.device,
) -> int:
    valid = np.where(mask)[0]
    if len(valid) == 0:
        return 0
    if random.random() < epsilon:
        return int(np.random.choice(valid))
    t = torch.FloatTensor(boards[valid]).unsqueeze(1).to(device)  # (n, 1, H, W)
    with torch.no_grad():
        q = q_net(t).cpu().numpy()
    return int(valid[int(np.argmax(q))])


def train_step(
    q_net: QNetwork,
    q_target: QNetwork,
    optimizer: optim.Optimizer,
    buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
) -> float | None:
    if len(buffer) < batch_size:
        return None

    afterstates, rewards, next_boards, next_masks, dones = buffer.sample(batch_size)
    B = batch_size
    H, W = afterstates.shape[1], afterstates.shape[2]

    # Current Q-values
    cur = torch.FloatTensor(afterstates).unsqueeze(1).to(device)   # (B, 1, H, W)
    q_pred = q_net(cur)                                             # (B,)

    # Double DQN: online net selects best next action, target net evaluates it
    nxt = torch.FloatTensor(next_boards).to(device)                 # (B, 40, H, W)
    nxt_flat = nxt.view(B * MAX_ACTIONS, 1, H, W)                   # (B*40, 1, H, W)
    mask_f = torch.FloatTensor(next_masks).to(device)               # (B, 40)

    with torch.no_grad():
        q_online = q_net(nxt_flat).view(B, MAX_ACTIONS)
        q_online = q_online * mask_f + (1.0 - mask_f) * (-1e9)
        best_acts = q_online.argmax(dim=1, keepdim=True)            # (B, 1)

        q_tgt = q_target(nxt_flat).view(B, MAX_ACTIONS)
        max_next_q = q_tgt.gather(1, best_acts).squeeze(1)         # (B,)

    dones_f = torch.FloatTensor(dones).to(device)
    max_next_q = max_next_q * (1.0 - dones_f)
    targets = torch.FloatTensor(rewards).to(device) + gamma * max_next_q

    loss = nn.MSELoss()(q_pred, targets.detach())
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
    optimizer.step()
    return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg["device"])
    print(f"Device: {device}")

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])

    q_net    = QNetwork().to(device)
    q_target = copy.deepcopy(q_net)
    q_target.eval()
    print(f"Parameters: {sum(p.numel() for p in q_net.parameters()):,}")

    optimizer = optim.Adam(q_net.parameters(), lr=cfg["learning_rate"])
    buffer    = ReplayBuffer(cfg["buffer_size"])

    os.makedirs("checkpoints", exist_ok=True)

    eps       = cfg["epsilon_start"]
    eps_start = cfg["epsilon_start"]
    eps_end   = cfg["epsilon_end"]
    eps_steps = cfg["epsilon_decay_steps"]

    total_steps = 0
    episode     = 0
    best_lines  = 0

    recent_rewards: list[float] = []
    recent_lines:   list[int]   = []
    recent_lengths: list[int]   = []

    while total_steps < cfg["total_steps"]:
        game = Game(seed=None)
        ep_reward = 0.0
        ep_lines  = 0
        ep_steps  = 0

        boards, _, mask = compute_afterstates(game)

        while not game.game_over and mask.any():
            action       = select_action(q_net, boards, mask, eps, device)
            chosen_board = boards[action].copy()

            rot, col  = action // 10, action % 10
            info      = game.place_piece(rot, col, use_hold=False)
            lines     = info.get("lines", 0)
            game_over = info.get("game_over", False) or game.game_over

            r          = reward_fn(lines, game_over)
            ep_reward += r
            ep_lines  += lines
            ep_steps  += 1
            total_steps += 1

            if not game_over:
                next_boards, _, next_mask = compute_afterstates(game)
            else:
                next_boards = np.zeros_like(boards)
                next_mask   = np.zeros(MAX_ACTIONS, dtype=np.float32)

            buffer.push(chosen_board, r, next_boards, next_mask, game_over)
            boards, mask = next_boards, next_mask.astype(bool)

            eps = max(eps_end, eps_start - (eps_start - eps_end) * total_steps / eps_steps)

            if total_steps % cfg["train_freq"] == 0:
                train_step(q_net, q_target, optimizer, buffer,
                           cfg["batch_size"], cfg["gamma"], device)

            if total_steps % cfg["target_update_freq"] == 0:
                q_target.load_state_dict(q_net.state_dict())

            if total_steps % cfg["checkpoint_freq"] == 0:
                torch.save(q_net.state_dict(), f"checkpoints/model_{total_steps}.pt")
                print(f"  Checkpoint saved at {total_steps:,} steps")

        episode += 1
        recent_rewards.append(ep_reward)
        recent_lines.append(ep_lines)
        recent_lengths.append(ep_steps)

        if ep_lines > best_lines:
            best_lines = ep_lines
            torch.save(q_net.state_dict(), "checkpoints/best_model.pt")

        if episode % cfg["log_freq"] == 0:
            n = len(recent_rewards)
            print(
                f"Ep {episode:6,} | Steps {total_steps:>10,} | "
                f"Reward {sum(recent_rewards)/n:7.1f} | "
                f"Lines {sum(recent_lines)/n:5.1f} | "
                f"Len {sum(recent_lengths)/n:5.1f} | "
                f"Eps {eps:.3f}"
            )
            recent_rewards.clear()
            recent_lines.clear()
            recent_lengths.clear()

    torch.save(q_net.state_dict(), "checkpoints/final_model.pt")
    print("Training complete.")


if __name__ == "__main__":
    main()
