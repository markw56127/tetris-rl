#!/usr/bin/env python3
"""Run a trained agent and optionally render with pygame."""
import argparse
import time

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from env.tetris_env import TetrisEnv
from env.wrappers import RecordEpisodeStatistics


def run_episode(model: MaskablePPO, env, render: bool = False) -> dict:
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
        obs, _, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        if render:
            env.render()
            time.sleep(0.05)
    return info.get("episode", info)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to .zip checkpoint")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    render_mode = "human" if args.render else None
    base_env = TetrisEnv(seed=args.seed, render_mode=render_mode)
    env = RecordEpisodeStatistics(base_env)
    env = ActionMasker(env, lambda e: e.unwrapped.action_masks())

    model = MaskablePPO.load(args.checkpoint)

    rewards, lines, lengths = [], [], []
    for ep in range(args.episodes):
        info = run_episode(model, env, render=args.render)
        r = info.get("r", info.get("episode", {}).get("r", 0))
        l = info.get("lines", info.get("episode", {}).get("lines", 0))
        le = info.get("l", info.get("episode", {}).get("l", 0))
        rewards.append(r)
        lines.append(l)
        lengths.append(le)
        print(f"Episode {ep+1:3d}: reward={r:7.1f}  lines={l:4d}  length={le:5d}")

    print(f"\nMean reward: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"Mean lines:  {np.mean(lines):.1f} ± {np.std(lines):.1f}")
    print(f"Mean length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")

    env.close()


if __name__ == "__main__":
    main()
