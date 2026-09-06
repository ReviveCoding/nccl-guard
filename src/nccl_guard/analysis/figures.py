from __future__ import annotations

import json
from pathlib import Path

FIGURES = [
    "topology_summary",
    "nccl_latency_busbw",
    "strong_scaling",
    "parallelism_ecr",
    "tuning_gap",
    "fault_recovery",
]


def generate_figures(root: Path) -> Path:
    out = root / "reports/figures/status.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({name: "PENDING_NO_EVIDENCE" for name in FIGURES}, indent=2) + "\n",
        encoding="utf-8",
    )
    return out
