from __future__ import annotations
from typing import Dict, List
import copy

from rl_hh.env import SchedIdenticalHHEnv


def run_greedy_hh(env: SchedIdenticalHHEnv, episodes: int = 20) -> List[Dict]:
    """
    Greedy hyperheuristic: at each step, try each operator once (on current state),
    pick the one yielding best immediate makespan, apply it.
    """
    rows: List[Dict] = []

    for ep in range(episodes):
        obs, info = env.reset()
        best = info["best_cmax"]
        lb = info["lb"]

        done = False
        while not done:
            # brute force choose best operator on a copy of schedule
            assert env.schedule is not None
            cur = env.schedule
            best_action = 0
            best_cmax = cur.makespan

            for a, op in enumerate(env.operators):
                cand = op.fn(cur, env.rng)
                if cand.makespan < best_cmax:
                    best_cmax = cand.makespan
                    best_action = a

            obs, r, terminated, truncated, info = env.step(best_action)
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