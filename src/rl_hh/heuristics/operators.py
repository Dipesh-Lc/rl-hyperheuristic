from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple
import numpy as np

from rl_hh.vendor.identical_scheduling import Schedule


def _clone_machines(machines: List[List[float]]) -> List[List[float]]:
    return [list(ms) for ms in machines]


def op_swap_two_jobs(schedule: Schedule, rng: np.random.Generator) -> Schedule:
    """
    Swap one random job between two (possibly different) machines.
    """
    machines = _clone_machines(schedule.machines)
    m = len(machines)
    if m < 1:
        return schedule

    nonempty = [i for i in range(m) if len(machines[i]) > 0]
    if len(nonempty) < 1:
        return schedule

    i = int(rng.choice(nonempty))
    j = int(rng.choice(nonempty)) if len(nonempty) > 1 else i

    ai = int(rng.integers(0, len(machines[i])))
    aj = int(rng.integers(0, len(machines[j])))

    machines[i][ai], machines[j][aj] = machines[j][aj], machines[i][ai]
    return Schedule(machines=machines)

def op_swap_best_of_k(schedule: Schedule, rng: np.random.Generator, k: int = 20) -> Schedule:
    best = schedule
    best_cmax = schedule.makespan
    for _ in range(k):
        cand = op_swap_two_jobs(schedule, rng)
        if cand.makespan < best_cmax:
            best, best_cmax = cand, cand.makespan
    return best

def op_move_max_to_min(schedule: Schedule, rng: np.random.Generator) -> Schedule:
    """
    Move one random job from the max-loaded machine to the min-loaded machine.
    """
    machines = _clone_machines(schedule.machines)
    if not machines:
        return schedule

    loads = [sum(ms) for ms in machines]
    i_max = int(np.argmax(loads))
    i_min = int(np.argmin(loads))

    if i_max == i_min or len(machines[i_max]) == 0:
        return schedule

    k = int(rng.integers(0, len(machines[i_max])))
    job = machines[i_max].pop(k)
    machines[i_min].append(job)
    return Schedule(machines=machines)


def op_two_smallest_from_max(schedule: Schedule, rng: np.random.Generator) -> Schedule:
    """
    Take up to two smallest jobs from max-loaded machine and move them to the currently min-loaded machine.
    """
    machines = _clone_machines(schedule.machines)
    if not machines:
        return schedule

    loads = [sum(ms) for ms in machines]
    i_max = int(np.argmax(loads))
    if len(machines[i_max]) == 0:
        return schedule

    # pick up to two smallest
    jobs = machines[i_max]
    idx_sorted = sorted(range(len(jobs)), key=lambda t: jobs[t])
    take = idx_sorted[: min(2, len(idx_sorted))]

    # remove in descending index order so indices remain valid
    moved = []
    for idx in sorted(take, reverse=True):
        moved.append(machines[i_max].pop(idx))

    # recompute min machine after removals
    loads2 = [sum(ms) for ms in machines]
    i_min = int(np.argmin(loads2))

    for job in moved:
        machines[i_min].append(job)

    return Schedule(machines=machines)


def op_local_search_k(schedule: Schedule, rng: np.random.Generator, k: int = 20) -> Schedule:
    """
    Try k random move candidates (max->min or random swap) and take the best improvement.
    """
    best = schedule
    best_cmax = schedule.makespan

    for _ in range(k):
        cand = schedule
        if rng.random() < 0.6:
            cand = op_move_max_to_min(schedule, rng)
        else:
            cand = op_swap_best_of_k(schedule, rng, k=10)

        cmax = cand.makespan
        if cmax < best_cmax:
            best, best_cmax = cand, cmax

    return best


def op_ruin_recreate(schedule: Schedule, rng: np.random.Generator, r: int = 5) -> Schedule:
    """
    Remove r random jobs globally, then greedily reinsert to least-loaded machine.
    """
    machines = _clone_machines(schedule.machines)
    m = len(machines)
    if m == 0:
        return schedule

    # collect all jobs with (machine, idx)
    all_pos: List[Tuple[int, int]] = []
    for i in range(m):
        for j in range(len(machines[i])):
            all_pos.append((i, j))
    if not all_pos:
        return schedule

    r = min(r, len(all_pos))
    chosen = rng.choice(len(all_pos), size=r, replace=False)

    # remove chosen jobs (sort by machine then descending idx)
    to_remove = [all_pos[int(c)] for c in chosen]
    to_remove.sort(key=lambda x: (x[0], -x[1]))

    removed: List[float] = []
    for i, j in to_remove:
        removed.append(machines[i].pop(j))

    # greedy reinsert
    loads = [sum(ms) for ms in machines]
    for job in removed:
        i_min = int(np.argmin(loads))
        machines[i_min].append(job)
        loads[i_min] += job

    return Schedule(machines=machines)


# Operator registry (discrete action set)

@dataclass(frozen=True)
class Operator:
    name: str
    fn: Callable[[Schedule, np.random.Generator], Schedule]


def default_operators() -> List[Operator]:
    return [
        Operator("move_max_to_min", op_move_max_to_min),
        Operator("two_smallest_from_max", op_two_smallest_from_max),

        Operator("swap_best_k10", lambda s, r: op_swap_best_of_k(s, r, k=10)),
        Operator("swap_best_k20", lambda s, r: op_swap_best_of_k(s, r, k=20)),

        Operator("local_search_k10", lambda s, r: op_local_search_k(s, r, k=10)),
        Operator("local_search_k20", lambda s, r: op_local_search_k(s, r, k=20)),
    ]