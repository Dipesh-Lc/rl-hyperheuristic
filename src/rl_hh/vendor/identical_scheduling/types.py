from __future__ import annotations
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class Schedule:
    machines: List[List[float]]  # each is a list of processing times

    @property
    def loads(self) -> List[float]:
        return [sum(ms) for ms in self.machines]

    @property
    def makespan(self) -> float:
        return max(self.loads) if self.machines else 0.0