from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass(frozen=True)
class IdenticalSchedSpec:
    n: int
    m: int
    dist: str
    seed: int

def generate_jobs(spec: IdenticalSchedSpec) -> List[float]:
    rng = np.random.default_rng(spec.seed)
    n = spec.n

    if spec.dist == "uniform":
        p = rng.uniform(1.0, 100.0, n)

    elif spec.dist == "bimodal":
        n1 = n // 2
        n2 = n - n1
        small = rng.uniform(1.0, 20.0, n1)
        large = rng.uniform(50.0, 100.0, n2)
        p = np.concatenate([small, large])
        p = p[rng.permutation(n)]

    elif spec.dist == "heavy_tail":
        # lognormal-ish
        p = rng.lognormal(mean=2.5, sigma=1.0, size=n)
        p = np.clip(p, 1.0, 200.0)

    else:
        raise ValueError(f"Unknown dist: {spec.dist}")

    return [float(x) for x in p]