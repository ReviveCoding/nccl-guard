from __future__ import annotations

from dataclasses import dataclass
from statistics import median

Interval = tuple[float, float]


def interval_union(intervals: list[Interval]) -> list[Interval]:
    clean = sorted((a, b) for a, b in intervals if b > a)
    merged: list[Interval] = []
    for start, end in clean:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def duration(intervals: list[Interval]) -> float:
    return sum(b - a for a, b in interval_union(intervals))


def exposed_communication(comm: list[Interval], compute: list[Interval]) -> float:
    pieces = interval_union(comm)
    for ks, ke in interval_union(compute):
        next_pieces: list[Interval] = []
        for start, end in pieces:
            if ke <= start or ks >= end:
                next_pieces.append((start, end))
            else:
                if start < ks:
                    next_pieces.append((start, ks))
                if ke < end:
                    next_pieces.append((ke, end))
        pieces = next_pieces
    return duration(pieces)


def ecr(comm: list[Interval], compute: list[Interval], update_duration: float) -> float:
    if update_duration <= 0:
        raise ValueError("update duration must be positive")
    return exposed_communication(comm, compute) / update_duration


def rank_imbalance(update_times: list[float]) -> float:
    if not update_times or median(update_times) <= 0:
        raise ValueError("positive rank times required")
    return (max(update_times) - median(update_times)) / median(update_times)


@dataclass(frozen=True)
class CollectiveRegime:
    collective: str
    ranks: int
    datatype: str
    message_min: int
    message_max: int
    nodes: int
    topology: str

    def comparable(self, other: CollectiveRegime) -> bool:
        return self == other
