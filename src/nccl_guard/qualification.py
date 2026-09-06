from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nccl_guard.analysis.statistics import summarize_launches
from nccl_guard.benchmark import BENCHMARK_SEMANTICS_VERSION, run_single
from nccl_guard.evidence.writer import verify_run
from nccl_guard.status import Status


def run_dense_launch_blocks(
    root: Path, *, launches: int = 3, warmup: int = 10, measured: int = 50
) -> tuple[Status, Path]:
    """Run/resume independent launch blocks without overwriting finalized evidence."""
    output = root / "evidence/local_dense_launch_blocks.json"
    records: list[dict[str, Any]] = []
    if output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if (
            previous.get("warmup_updates") == warmup
            and previous.get("measured_updates") == measured
            and previous.get("benchmark_semantics_version") == BENCHMARK_SEMANTICS_VERSION
        ):
            for record in previous.get("launches", []):
                run_path = root / "artifacts" / record["run_id"]
                if run_path.is_dir() and verify_run(run_path):
                    records.append(record)

    while len(records) < launches:
        status, run_path, metrics = run_single(root, warmup=warmup, measured=measured)
        record = {
            "run_id": run_path.name,
            "status": status,
            "tokens_per_second": metrics.get("tokens_per_second"),
            "update_time_mean_s": metrics.get("update_time_mean_s"),
            "temperature_before_after": [
                metrics.get("telemetry_before"),
                metrics.get("telemetry_after"),
            ],
        }
        records.append(record)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "status": "INCOMPLETE_RESUMABLE",
                    "warmup_updates": warmup,
                    "measured_updates": measured,
                    "benchmark_semantics_version": BENCHMARK_SEMANTICS_VERSION,
                    "launches": records,
                },
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        if status is not Status.PASS:
            return status, output

    values = [float(record["tokens_per_second"]) for record in records]
    value = {
        "status": "PASS",
        "warmup_updates": warmup,
        "measured_updates": measured,
        "benchmark_semantics_version": BENCHMARK_SEMANTICS_VERSION,
        "launches": records,
        "tokens_per_second": summarize_launches(values),
        "statistical_unit": "independent process/function launch block",
        "claim_note": "single-configuration characterization; not a comparative speed claim",
    }
    output.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    return Status.PASS, output
