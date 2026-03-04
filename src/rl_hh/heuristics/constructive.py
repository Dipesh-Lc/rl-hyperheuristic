from __future__ import annotations
from typing import List, Sequence
import numpy as np

from rl_hh.vendor.identical_scheduling import Schedule, list_scheduling, lpt


def random_assignment(p: Sequence[float], m: int, rng: np.random.Generator) -> Schedule:
    if m <= 0:
        raise ValueError("m must be >= 1")

    machines: List[List[float]] = [[] for _ in range(m)]
    for pj in p:
        i = int(rng.integers(0, m))
        machines[i].append(float(pj))
    return Schedule(machines=machines)


def greedy_list(p: Sequence[float], m: int) -> Schedule:
    return list_scheduling(p, m)


def lpt_construct(p: Sequence[float], m: int) -> Schedule:
    return lpt(p, m)