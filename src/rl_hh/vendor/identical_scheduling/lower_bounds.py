from __future__ import annotations
import math
from typing import Sequence

def lb_makespan_identical(p: Sequence[float], m: int) -> float:
    if m <= 0:
        raise ValueError("m must be >= 1")
    if len(p) == 0:
        return 0.0
    total = sum(p)
    return max(total / m, max(p))