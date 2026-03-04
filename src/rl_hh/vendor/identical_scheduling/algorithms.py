from __future__ import annotations
from typing import List, Sequence
import heapq

from .types import Schedule

def list_scheduling(p: Sequence[float], m: int) -> Schedule:
    """
    Graham's list scheduling for identical machines:
    assign each job to currently least-loaded machine.
    Uses a heap for O(n log m).
    """
    if m <= 0:
        raise ValueError("m must be >= 1")
    machines: List[List[float]] = [[] for _ in range(m)]

    heap = [(0.0, i) for i in range(m)]
    heapq.heapify(heap)

    for pj in p:
        load, i = heapq.heappop(heap)
        machines[i].append(float(pj))
        load += float(pj)
        heapq.heappush(heap, (load, i))

    return Schedule(machines=machines)

def lpt(p: Sequence[float], m: int) -> Schedule:
    """
    LPT: sort jobs by decreasing processing time, then list scheduling.
    """
    p_sorted = sorted((float(x) for x in p), reverse=True)
    return list_scheduling(p_sorted, m=m)