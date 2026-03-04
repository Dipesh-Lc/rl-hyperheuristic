from __future__ import annotations
from dataclasses import asdict
from typing import Optional
import os
import time

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from rl_hh.env import SchedIdenticalHHEnv, EnvConfig
from rl_hh.utils.seed import set_seed

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(8)
torch.set_float32_matmul_precision("high")


def make_env(rank: int, *, n: int, m: int, dist: str, seed: int, max_steps: int,
             init_solution: str, step_penalty: float):
    def _init():
        local_cfg = EnvConfig(
            n=n,
            m=m,
            dist=dist,
            seed=seed + rank,
            max_steps=max_steps,
            init_solution=init_solution,
            step_penalty=step_penalty,
        )
        return Monitor(SchedIdenticalHHEnv(local_cfg))
    return _init

def train_dqn(
    out_dir: str = "results",
    seed: int = 0,
    n: int = 40,
    m: int = 4,
    dist: str = "uniform",
    max_steps: int = 200,
    total_timesteps: int = 200_000,
    init_solution: str = "random",
    step_penalty: float = 0.0001,
    load_path: str | None = None
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)

    cfg = EnvConfig(n=n, m=m, dist=dist, seed=seed, max_steps=max_steps, init_solution=init_solution, step_penalty=step_penalty)

    n_envs = 12
    env = SubprocVecEnv([
        make_env(
            i,
            n=n, m=m, dist=dist, seed=seed,
            max_steps=max_steps,
            init_solution=init_solution,
            step_penalty=step_penalty,
        )
        for i in range(n_envs)
    ])

    # Auto-scale replay parameters with max_steps to ensure enough learning starts and buffer size for longer episodes
    learning_starts = 50 * max_steps
    buffer_size = 10 * learning_starts

    # Scale batch size with problem size
    if n <= 60:
        batch_size = 256
    elif n <= 200:
        batch_size = 512
    else:
        batch_size = 1024

    if load_path is not None:
        # Resume training from an existing model
        model = DQN.load(load_path, env=env)
        model.set_env(env)
        model.verbose = 1
    else:
        # Fresh model
        model = DQN(
            policy="MlpPolicy",
            env=env,
            device=device,
            learning_rate=3e-4,
            learning_starts=learning_starts,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=1.0,
            gamma=0.99,
            train_freq=128,
            target_update_interval=2000,
            exploration_fraction=0.6,
            exploration_final_eps=0.1,
            verbose=1,
            tensorboard_log=os.path.join(out_dir, "tb"),
            seed=seed,
        )

    model.learn(total_timesteps=total_timesteps)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(out_dir, f"dqn_sched_identical_n{n}_m{m}_{dist}_seed{seed}_{run_id}.zip")
    model.save(model_path)

    # save config used
    cfg_path = os.path.join(out_dir, f"train_config_{run_id}.txt")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(str(asdict(cfg)) + "\n")
        f.write(f"total_timesteps={total_timesteps}\n")
        f.write(f"model_path={model_path}\n")

    env.close()
    return model_path