"""
Thin wrapper around sb3-contrib MaskablePPO that wires in our CNN extractor.
Direct training entry point is train.py; this module handles model construction.
"""
from __future__ import annotations
from typing import Any, Dict

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from agent.networks import TetrisCNNExtractor
from env.wrappers import make_env


def _make_env_fn(seed: int):
    def _init():
        env = make_env(seed=seed, record_stats=True)
        # Wrap so sb3-contrib can call action_masks()
        env = ActionMasker(env, lambda e: e.unwrapped.action_masks())
        return env
    return _init


def build_model(
    n_envs: int = 8,
    seed: int = 0,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 256,
    n_epochs: int = 4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    device: str = "auto",
    tensorboard_log: str = "runs/",
) -> MaskablePPO:
    env_fns = [_make_env_fn(seed + i) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    policy_kwargs: Dict[str, Any] = {
        "features_extractor_class": TetrisCNNExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": [],  # features extractor already has full architecture
    }

    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        seed=seed,
        device=device,
        tensorboard_log=tensorboard_log,
        verbose=1,
        policy_kwargs=policy_kwargs,
    )
    return model


def load_model(path: str, n_envs: int = 1, seed: int = 0) -> MaskablePPO:
    env_fns = [_make_env_fn(seed + i) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)
    model = MaskablePPO.load(path, env=vec_env)
    return model
