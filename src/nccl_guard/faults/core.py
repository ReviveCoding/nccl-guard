from __future__ import annotations

from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True)
class FaultEvent:
    step: int
    rank: int
    kind: str
    duration_ms: float = 0.0

    def validate(self) -> None:
        if (
            self.step < 0
            or self.rank < 0
            or self.kind not in {"compute_delay", "workload_skew", "rank_failure"}
        ):
            raise ValueError("invalid fault event")


def robust_straggler(history: list[float], value: float, k: float = 6.0) -> bool:
    if len(history) < 5:
        return False
    center = median(history)
    mad = median(abs(x - center) for x in history)
    return value > center + k * max(mad, 1e-9)
