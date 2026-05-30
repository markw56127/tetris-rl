#!/usr/bin/env python3
import argparse
import numpy as np
import torch

from tetris.game import Game
from env.afterstate import compute_afterstates, MAX_ACTIONS
from agent.networks import QNetwork


def evaluate(model_path: str, n_episodes: int = 20, device_name: str = "cpu") -> None:
    device = torch.device(device_name)
    q_net = QNetwork().to(device)
    q_net.load_state_dict(torch.load(model_path, map_location=device))
    q_net.eval()

    all_lines, all_lengths = [], []

    for ep in range(n_episodes):
        game = Game(seed=ep)
        ep_lines = 0
        ep_steps = 0

        boards, _, mask, queue = compute_afterstates(game)

        while not game.game_over and mask.any():
            valid = np.where(mask)[0]
            b = torch.FloatTensor(boards[valid]).unsqueeze(1).to(device)
            q = torch.LongTensor(queue).unsqueeze(0).expand(len(valid), -1).to(device)
            with torch.no_grad():
                vals = q_net(b, q).cpu().numpy()
            action = int(valid[int(np.argmax(vals))])

            rot, col = action // 10, action % 10
            info = game.place_piece(rot, col, use_hold=False)
            ep_lines += info.get("lines", 0)
            ep_steps += 1

            if not game.game_over:
                boards, _, mask, queue = compute_afterstates(game)

        all_lines.append(ep_lines)
        all_lengths.append(ep_steps)
        print(f"  Ep {ep+1:3d}: {ep_steps:4d} pieces, {ep_lines:3d} lines")

    print(f"\nResults over {n_episodes} episodes:")
    print(f"  Lines cleared: {np.mean(all_lines):.1f} ± {np.std(all_lines):.1f}")
    print(f"  Pieces placed: {np.mean(all_lengths):.1f} ± {np.std(all_lengths):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/best_model.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    evaluate(args.model, args.episodes, args.device)
