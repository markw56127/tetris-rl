#!/usr/bin/env python3
import argparse
import os
import sys

import yaml
import numpy as np
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    BaseCallback,
)
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.evaluation import evaluate_policy as maskable_evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor

from agent.ppo import build_model, _make_env_fn
from env.wrappers import make_env


class MaskableEvalCallback(BaseCallback):
    """Eval callback that passes action masks to MaskablePPO.predict."""

    def __init__(self, eval_env, n_eval_episodes=20, eval_freq=10000,
                 best_model_path=None, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_model_path = best_model_path
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            rewards, lengths = maskable_evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
                return_episode_rewards=True,
            )
            mean_r, std_r = np.mean(rewards), np.std(rewards)
            mean_l = np.mean(lengths)
            if self.verbose:
                print(f"Eval ({self.num_timesteps:,} steps): "
                      f"reward={mean_r:.2f} ± {std_r:.2f}  length={mean_l:.1f}")
            self.logger.record("eval/mean_reward", mean_r)
            self.logger.record("eval/mean_length", mean_l)
            if mean_r > self.best_mean_reward and self.best_model_path:
                self.best_mean_reward = mean_r
                self.model.save(self.best_model_path)
                if self.verbose:
                    print(f"  New best model saved ({mean_r:.2f})")
        return True


def linear_schedule(initial: float):
    def schedule(progress: float) -> float:
        return initial * (0.1 + 0.9 * progress)  # decays from initial to 0.1 * initial
    return schedule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]
    ppo_cfg = cfg["ppo"]

    os.makedirs(train_cfg["log_dir"], exist_ok=True)
    os.makedirs(train_cfg["checkpoint_dir"], exist_ok=True)

    lr = ppo_cfg["learning_rate"]
    if ppo_cfg.get("lr_schedule") == "linear":
        lr = linear_schedule(ppo_cfg["learning_rate"])

    if args.resume:
        from agent.ppo import load_model
        # SB3 appends .zip automatically; strip it if already present to avoid double suffix
        resume_path = args.resume
        if resume_path.endswith(".zip"):
            resume_path = resume_path[:-4]
        print(f"Resuming from {resume_path}")
        model = load_model(resume_path, n_envs=train_cfg["n_envs"], seed=train_cfg["seed"])
    else:
        model = build_model(
            n_envs=train_cfg["n_envs"],
            seed=train_cfg["seed"],
            learning_rate=lr,
            n_steps=ppo_cfg["n_steps"],
            batch_size=ppo_cfg["batch_size"],
            n_epochs=ppo_cfg["n_epochs"],
            gamma=ppo_cfg["gamma"],
            gae_lambda=ppo_cfg["gae_lambda"],
            clip_range=ppo_cfg["clip_range"],
            ent_coef=ppo_cfg["ent_coef"],
            vf_coef=ppo_cfg["vf_coef"],
            max_grad_norm=ppo_cfg["max_grad_norm"],
            device=ppo_cfg["device"],
            tensorboard_log=train_cfg["log_dir"],
        )

    # Eval env: DummyVecEnv avoids subprocess overhead for a single eval worker
    eval_env_fn = _make_env_fn(train_cfg["seed"] + 9999)
    eval_env = DummyVecEnv([eval_env_fn])
    eval_env = VecMonitor(eval_env)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(train_cfg["checkpoint_freq"] // train_cfg["n_envs"], 1),
        save_path=train_cfg["checkpoint_dir"],
        name_prefix="tetris_ppo",
        verbose=1,
    )
    eval_cb = MaskableEvalCallback(
        eval_env,
        n_eval_episodes=train_cfg["eval_episodes"],
        eval_freq=max(train_cfg["eval_freq"] // train_cfg["n_envs"], 1),
        best_model_path=os.path.join(train_cfg["checkpoint_dir"], "best_model"),
        verbose=1,
    )

    model.learn(
        total_timesteps=train_cfg["total_timesteps"],
        callback=CallbackList([checkpoint_cb, eval_cb]),
    )
    model.save(os.path.join(train_cfg["checkpoint_dir"], "final_model"))
    print("Training complete. Model saved.")


if __name__ == "__main__":
    main()
