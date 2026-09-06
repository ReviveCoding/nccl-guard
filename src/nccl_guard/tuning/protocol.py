from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    name: str
    strategy: str
    microbatch: int
    grad_accumulation: int
    tp_degree: int = 1


@dataclass(frozen=True)
class Observation:
    candidate: str
    throughput: float
    pool: str
    correctness: bool = True
    oom: bool = False
    p95_regression: float = 0.0


class SearchView:
    """A capability object that cannot expose evaluation observations."""

    def __init__(self, observations: Iterable[Observation]) -> None:
        rows = list(observations)
        if any(row.pool != "SEARCH" for row in rows):
            raise ValueError("planner/search view accepts SEARCH observations only")
        self._rows = rows

    def outcome(self, name: str) -> Observation | None:
        return next((row for row in self._rows if row.candidate == name), None)

    def names(self) -> list[str]:
        return [row.candidate for row in self._rows]


def eligible(row: Observation) -> bool:
    return row.correctness and not row.oom and row.p95_regression <= 0.10


def exhaustive_reference(view: SearchView) -> str:
    rows = [view.outcome(name) for name in view.names()]
    valid = [row for row in rows if row is not None and eligible(row)]
    if not valid:
        raise ValueError("no guarded candidate")
    return max(valid, key=lambda row: row.throughput).candidate


def random_search(candidates: list[Candidate], view: SearchView, budget: int, seed: int) -> str:
    chosen = random.Random(seed).sample(candidates, min(budget, len(candidates)))
    subset = SearchView(
        row for candidate in chosen if (row := view.outcome(candidate.name)) is not None
    )
    return exhaustive_reference(subset)


def tpe_search(candidates: list[Candidate], view: SearchView, budget: int, seed: int) -> str:
    # Optuna is optional; deterministic offline replay avoids multiplying GPU runs.
    try:
        import optuna
    except ImportError:
        return random_search(candidates, view, budget, seed)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    names = [candidate.name for candidate in candidates]
    visited: list[str] = []
    target = min(budget, len(candidates))
    # A repeated TPE suggestion is not a new configuration evaluation. Keep asking over the
    # stable categorical space until the declared number of unique evaluations is reached.
    attempts = 0
    while len(set(visited)) < target and attempts < max(20, target * 20):
        trial = study.ask()
        name = trial.suggest_categorical("candidate", names)
        if name not in visited:
            visited.append(name)
        row = view.outcome(name)
        study.tell(trial, row.throughput if row is not None and eligible(row) else float("-inf"))
        attempts += 1
    if len(visited) < target:
        remaining = [name for name in names if name not in visited]
        random.Random(seed).shuffle(remaining)
        visited.extend(remaining[: target - len(visited)])

    def score(name: str) -> float:
        row = view.outcome(name)
        return row.throughput if row is not None and eligible(row) else float("-inf")

    return max(set(visited), key=score)


class GuardPlanner:
    def __init__(self, candidate_budget: int = 3) -> None:
        if candidate_budget > 3:
            raise ValueError("v1 primary Guard budget is at most 3")
        self.budget = candidate_budget

    def propose(
        self, candidates: list[Candidate], hardware: dict[str, object], diagnosis: str
    ) -> list[Candidate]:
        valid = [
            c for c in candidates if not hardware.get("native_windows") or c.strategy == "single"
        ]
        if diagnosis.startswith("COMM"):
            valid.sort(key=lambda c: (c.tp_degree, c.microbatch))
        else:
            valid.sort(key=lambda c: (-c.microbatch, c.grad_accumulation))
        return valid[: self.budget]

    def select(self, proposed: list[Candidate], view: SearchView) -> str:
        allowed = SearchView(row for c in proposed if (row := view.outcome(c.name)) is not None)
        return exhaustive_reference(allowed)
