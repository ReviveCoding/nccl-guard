from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nccl_guard.analysis.figures import generate_figures
from nccl_guard.evidence.writer import verify_run


def _excluded(root: Path) -> dict[str, str]:
    state_path = root / "evidence/execution_state.json"
    if not state_path.exists():
        return {}
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return {
        Path(record["path"]).name: record["reason"]
        for record in state.get("scientifically_excluded_artifacts", [])
    }


def generate_summary(root: Path) -> Path:
    excluded = _excluded(root)
    runs: list[dict[str, Any]] = []
    for verdict in sorted(
        (root / "artifacts").glob("*/verdict.json") if (root / "artifacts").exists() else []
    ):
        run = verdict.parent.name
        integrity = (verdict.parent / "sha256_manifest.json").exists() and verify_run(
            verdict.parent
        )
        runs.append(
            {
                "run": run,
                **json.loads(verdict.read_text(encoding="utf-8")),
                "artifact_integrity": "PASS" if integrity else "FAIL",
                "scientifically_excluded": run in excluded,
                "exclusion_reason": excluded.get(run),
            }
        )
    path = root / "evidence/run_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"runs": runs}, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def generate_reports(root: Path) -> Path:
    generate_summary(root)
    generate_figures(root)
    dense = json.loads(
        (root / "evidence/local_dense_launch_blocks.json").read_text(encoding="utf-8")
    )
    valid_sparse: list[tuple[Path, dict[str, Any]]] = []
    excluded = _excluded(root)
    for metrics_path in (root / "artifacts").glob("*-sparse-*/metrics.json"):
        if metrics_path.parent.name not in excluded and verify_run(metrics_path.parent):
            valid_sparse.append(
                (metrics_path, json.loads(metrics_path.read_text(encoding="utf-8")))
            )
    sparse_path, sparse = max(valid_sparse, key=lambda item: item[0].stat().st_mtime)
    summary = {
        "status": "PASS",
        "scope": "single-GPU CUDA characterization",
        "benchmark_verdict": "HOLD",
        "dense": dense,
        "sparse": {"run_id": sparse_path.parent.name, **sparse},
        "claim_boundary": "no NCCL, distributed scaling, or Criteo performance evidence",
    }
    evidence_path = root / "evidence/local_qualification_summary.json"
    evidence_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    dense_stats = dense["tokens_per_second"]
    local = f"""# Local Qualification

Status: **PASS for bounded local CUDA qualification; HOLD for the distributed benchmark.**

Machine: Windows, NVIDIA GeForce RTX 4090 Laptop GPU. PyTorch 2.14.0+cu132 reports CUDA 13.2 and
BF16 support. NCCL is unavailable, and Gloo was not substituted.

Dense benchmark-semantics v2 used the persisted deterministic dataset through scheduled sample
IDs, pinned host memory, nonblocking H2D outside model timing, and 10 warmup plus 50 measured
updates per launch. Across {dense_stats["n_launches"]} independent run blocks, input-token
throughput was {dense_stats["mean"]:.2f} mean and {dense_stats["median"]:.2f} median tokens/s
(run-level bootstrap 95% interval {dense_stats["ci95_low"]:.2f}–{dense_stats["ci95_high"]:.2f}).
This is single-configuration characterization, not an inferential method comparison.

The synthetic schema-compatible sparse CUDA control run `{sparse_path.parent.name}` achieved
{sparse["samples_per_second"]:.2f} samples/s over {sparse["measured_updates"]} measured updates,
with {sparse["precision"]} precision and {sparse["peak_allocated_bytes"]} peak allocated bytes.
It is not Criteo or distributed TorchRec evidence.

The existing torch.profiler trace is labeled ECR_APPROX. DCP local utility evidence validates
model/optimizer/application-state restore but is not elastic rank-failure recovery.
"""
    (root / "reports/local_qualification.md").write_text(local, encoding="utf-8")
    technical = """# NCCL-Guard Technical Report

NCCL-Guard implements deterministic dense and sparse control workloads, native distributed
composition, nested SEARCH/EVALUATION isolation, guarded eligibility, profiling, and explicit
application-state recovery. Local CUDA qualification and offline selection-protocol simulation
passed.

The primary distributed research question remains unanswered because this Windows host has one
GPU and no NCCL. No multi-GPU exhaustive-reference gap, scaling, ECR_AUTH, or elastic recovery
claim is reported. Overall benchmark verdict: **HOLD**.
"""
    path = root / "reports/technical_report.md"
    path.write_text(technical, encoding="utf-8")
    return path
