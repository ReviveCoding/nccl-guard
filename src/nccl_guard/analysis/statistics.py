from __future__ import annotations

import random
import statistics


def summarize_launches(
    values: list[float], seed: int = 17, samples: int = 2000
) -> dict[str, float | int]:
    if not values:
        raise ValueError("independent launch values required")
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return {
        "n_launches": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "ci95_low": means[int(0.025 * samples)],
        "ci95_high": means[int(0.975 * samples)],
    }
