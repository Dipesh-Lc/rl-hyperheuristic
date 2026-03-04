from __future__ import annotations
from typing import Dict, List
import numpy as np

from rl_hh.env import SchedIdenticalHHEnv


def run_random_hh(env: SchedIdenticalHHEnv, episodes: int = 20) -> List[Dict]:
    rows: List[Dict] = []
    for ep in range(episodes):
        obs, info = env.reset()
        best = info["best_cmax"]
        lb = info["lb"]

        done = False
        while not done:
            a = int(env.action_space.sample())
            obs, r, terminated, truncated, info = env.step(a)
            best = min(best, info["best_cmax"])
            done = terminated or truncated

        rows.append(
            {
                "episode": ep,
                "best_cmax": best,
                "lb": lb,
                "best_ratio": best / (lb + 1e-12),
            }
        )
    return rows