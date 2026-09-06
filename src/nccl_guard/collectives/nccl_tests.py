from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OPERATIONS = {"all_reduce", "all_gather", "reduce_scatter", "alltoall"}


def safe_max_message_bytes(free_hbm: int, ranks_per_process: int = 1) -> int:
    if free_hbm <= 0 or ranks_per_process <= 0:
        raise ValueError("positive memory and ranks required")
    # nccl-tests allocates send/receive buffers; reserve 75% for runtime and fragmentation.
    return min(1 << 30, int(free_hbm * 0.25 / (2 * ranks_per_process)))


def command(operation: str, output: Path, free_hbm: int, gpus: int) -> list[str]:
    if operation not in OPERATIONS or gpus not in {2, 4, 8}:
        raise ValueError("invalid nccl-tests configuration")
    return [
        f"./external/nccl-tests/build/{operation}_perf",
        "-b",
        "8",
        "-e",
        str(safe_max_message_bytes(free_hbm, gpus)),
        "-f",
        "2",
        "-g",
        str(gpus),
        "-w",
        "10",
        "-n",
        "50",
        "-N",
        "3",
        "-c",
        "1",
        "-T",
        "180",
        "-J",
        str(output),
    ]


def parse_json(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("results", raw if isinstance(raw, list) else [])
    if not isinstance(rows, list):
        raise ValueError("nccl-tests JSON lacks result rows")
    return [dict(row) for row in rows]
