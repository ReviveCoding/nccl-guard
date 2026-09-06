from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import pandas as pd

from nccl_guard.config.core import PROFILES
from nccl_guard.data.core import SampleSchedule, verify_manifest, verify_no_duplicate_or_skip
from nccl_guard.data.external import verify_external
from nccl_guard.evidence.writer import RunWriter, verify_run
from nccl_guard.hardware.doctor import collect_capabilities
from nccl_guard.models.dense.transformer import canonical_model, state_fingerprint
from nccl_guard.status import Status

LOCAL = {"E00", "E02", "E05", "E06"}
LINUX_MULTI_GPU = {"E01", "E03", "E04", "E08", "E09", "E10", "E12"}


def _e00_checks(root: Path, caps: dict[str, Any]) -> dict[str, bool]:
    dense_path = root / "evidence/local_dense_launch_blocks.json"
    dense = json.loads(dense_path.read_text(encoding="utf-8")) if dense_path.exists() else {}
    dense_runs = dense.get("launches", [])
    external = verify_external(root)
    schedule = SampleSchedule(17, 4, 8)
    model = canonical_model(PROFILES["local"], 17)
    return {
        "cuda_available": bool(caps["gpu"]["available"]),
        "bf16_or_fp16": bool(caps.get("bf16")) or bool(caps["gpu"]["available"]),
        "synthetic_manifest": verify_manifest(root, root / "data/manifests/synthetic_dense.json"),
        "external_acquired_hashes": all(
            value["status"] != Status.FAIL for value in external.values()
        ),
        "dense_launch_blocks": dense.get("status") == "PASS"
        and len(dense_runs) >= 3
        and all(verify_run(root / "artifacts" / row["run_id"]) for row in dense_runs),
        "canonical_initial_state": state_fingerprint(model)
        == state_fingerprint(canonical_model(PROFILES["local"], 17)),
        "sample_schedule": verify_no_duplicate_or_skip(schedule, 4),
    }


def run_experiment(root: Path, experiment: str, *, dry_run: bool = False) -> tuple[Status, Path]:
    experiment = experiment.upper()
    if experiment not in {f"E{i:02d}" for i in range(13)}:
        return Status.INVALID_CONFIGURATION, root / "evidence"
    caps = collect_capabilities()
    status, reason = Status.PASS, "control-path qualification"
    decision: dict[str, object] = {"applicable": False}
    checks: dict[str, bool] = {}
    if experiment == "E00" and not dry_run:
        checks = _e00_checks(root, caps)
        status = Status.PASS if all(checks.values()) else Status.FAIL
        reason = "local environment/data/model/evidence qualification"
    elif experiment in LINUX_MULTI_GPU:
        if platform.system() == "Windows" or caps["gpu"]["count"] < 2:
            status, reason = (
                Status.SKIP_UNSUPPORTED,
                "requires Linux with at least two GPUs; Gloo is not substituted",
            )
        else:
            status, reason = (
                Status.FAIL,
                "metadata gate cannot qualify distributed execution; use the documented "
                "nccl-tests/torchrun launcher",
            )
    elif experiment == "E07":
        status, reason = (
            Status.SKIP_RESOURCE_BUDGET,
            "final bounded C4/Criteo workload not prepared",
        )
    elif experiment == "E11":
        status, reason = (
            Status.SKIP_NEEDS_CREDENTIALS,
            "TorchTitan isolated validation not configured",
        )
    elif experiment == "E05" and not dry_run:
        from nccl_guard.tuning.simulation import run_nested_simulation

        simulation = run_nested_simulation(root)
        reason = "deterministic offline nested SEARCH/EVALUATION protocol simulation"
        decision = {
            "applicable": True,
            "evidence_kind": "control-logic simulation",
            "source": str(simulation.relative_to(root)),
            "performance_claim": False,
            "simulation_result": json.loads(simulation.read_text(encoding="utf-8")),
        }
    elif experiment == "E06" and not dry_run:
        from nccl_guard.tuning.simulation import run_planner_ablation_simulation

        simulation = run_planner_ablation_simulation(root)
        reason = "deterministic offline planner-ablation simulation"
        decision = {
            "applicable": True,
            "evidence_kind": "control-logic simulation",
            "source": str(simulation.relative_to(root)),
            "performance_claim": False,
            "simulation_result": json.loads(simulation.read_text(encoding="utf-8")),
        }
    if dry_run and status is Status.PASS:
        reason = "dry-run preflight"
    with RunWriter(root, experiment) as writer:
        writer.json("environment.json", caps)
        writer.json("hardware.json", caps.get("gpu", {}))
        writer.json("topology.json", caps.get("topology"))
        writer.json(
            "software_versions.json",
            {"torch": caps.get("torch"), "cuda": caps.get("torch_cuda_build")},
        )
        writer.json("dataset_manifest.json", {"applicable": experiment in {"E00", "E02", "E07"}})
        writer.json("sample_schedule.json", {"applicable": experiment in LOCAL})
        writer.json("model_config.json", {"applicable": experiment != "E01"})
        writer.json("strategy_config.json", {"experiment": experiment, "dry_run": dry_run})
        writer.json("initial_state_fingerprint.json", {"applicable": experiment in LOCAL})
        writer.json(
            "correctness.json",
            {"checks": checks, "passed": all(checks.values()) if checks else None},
        )
        pd.DataFrame(columns=["update", "update_time_s"]).to_parquet(
            writer.path / "metrics.parquet", index=False
        )
        pd.DataFrame(columns=["rank", "update_time_s"]).to_parquet(
            writer.path / "rank_metrics.parquet", index=False
        )
        pd.DataFrame(columns=["collective", "duration_s"]).to_parquet(
            writer.path / "collective_metrics.parquet", index=False
        )
        writer.json("planner_decision.json", decision)
        writer.json("fault_events.json", [])
        writer.json("recovery.json", {"applicable": experiment == "E10"})
        writer.finalize({"status": status, "reason": reason, "performance_claim": False})
        return status, writer.path
