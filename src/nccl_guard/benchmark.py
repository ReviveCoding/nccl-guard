from __future__ import annotations

import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from nccl_guard.config.core import PROFILES
from nccl_guard.data.core import SampleSchedule
from nccl_guard.evidence.writer import RunWriter
from nccl_guard.hardware.doctor import collect_capabilities
from nccl_guard.models.dense.transformer import canonical_model, state_fingerprint
from nccl_guard.status import Status

BENCHMARK_SEMANTICS_VERSION = 2


def run_single(
    root: Path, device: str = "cuda", warmup: int = 3, measured: int = 10
) -> tuple[Status, Path, dict[str, Any]]:
    if device != "cuda" or not torch.cuda.is_available():
        with RunWriter(root, "single") as writer:
            writer.finalize(
                {"status": Status.FAIL_GPU_EXECUTION, "reason": "CUDA required; no CPU fallback"}
            )
            return Status.FAIL_GPU_EXECUTION, writer.path, {}
    caps_before = collect_capabilities()
    config = PROFILES["local"]
    torch.manual_seed(17)
    model = canonical_model(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    schedule = SampleSchedule(17, warmup + measured, 2)
    manifest_path = root / "data/manifests/synthetic_dense.json"
    data_path = root / "data/processed/synthetic_dense_v1.bin"
    if not manifest_path.exists() or not data_path.exists():
        raise FileNotFoundError("prepare and verify synthetic dense data before benchmarking")
    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = int(dataset_manifest["rows"])
    stored_sequence = int(dataset_manifest["tokens"]) // rows
    if stored_sequence != config.sequence_length + 1:
        raise ValueError("synthetic dataset sequence length does not match model profile")
    mapped_rows = [[sample_id % rows for sample_id in update] for update in schedule.ids()]
    mapped = np.memmap(data_path, mode="r", dtype=np.uint16, shape=(rows, stored_sequence))
    # Copy and pin all scheduled input before timing; no parsing or materialization enters a step.
    host_batches = [
        torch.from_numpy(np.array(mapped[row_ids], dtype=np.int64, copy=True)).pin_memory()
        for row_ids in mapped_rows
    ]
    times: list[float] = []
    h2d_times: list[float] = []
    losses: list[float] = []
    metrics: dict[str, Any]
    finite_gradients = True
    grad_norm = 0.0
    torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(warmup + measured):
            h2d_start = torch.cuda.Event(enable_timing=True)
            h2d_end = torch.cuda.Event(enable_timing=True)
            h2d_start.record()
            tokens = host_batches[step].cuda(non_blocking=True)
            h2d_end.record()
            torch.cuda.synchronize()
            x, y = tokens[:, :-1], tokens[:, 1:]
            optimizer.zero_grad(set_to_none=True)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            with torch.autocast("cuda", dtype=dtype):
                _, loss = model(x, y)
            assert loss is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            end.record()
            torch.cuda.synchronize()
            # Correctness inspection is intentionally outside the CUDA-event timing region.
            grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
            finite_gradients = finite_gradients and all(
                torch.isfinite(grad).all() for grad in grads
            )
            if step == warmup + measured - 1:
                squared_norm = torch.zeros((), device=x.device)
                for grad in grads:
                    squared_norm += grad.float().pow(2).sum()
                grad_norm = float(torch.sqrt(squared_norm))
            if step >= warmup:
                h2d_times.append(h2d_start.elapsed_time(h2d_end) / 1000)
                times.append(start.elapsed_time(end) / 1000)
                losses.append(float(loss.detach()))
    except torch.cuda.OutOfMemoryError as exc:
        status, metrics = Status.VALID_BUT_OOM, {"error": str(exc)}
    except Exception as exc:
        status, metrics = Status.FAIL_GPU_EXECUTION, {"error": repr(exc)}
    else:
        ordered = sorted(times)
        p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
        mean_time = statistics.mean(times)
        metrics = {
            "tokens_per_second": (2 * config.sequence_length) / mean_time,
            "samples_per_second": 2 / mean_time,
            "update_time_mean_s": mean_time,
            "h2d_time_mean_s": statistics.mean(h2d_times),
            "step_p50_s": statistics.median(times),
            "step_p95_s": p95,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "loss": losses[-1],
            "grad_norm": grad_norm,
            "parameter_count": model.parameter_count(),
            "precision": precision,
            "autocast": True,
            "grad_scaler": scaler.is_enabled(),
            "device": str(next(model.parameters()).device),
            "compile": False,
            "warmup_updates": warmup,
            "measured_updates": measured,
            "input_mode": "host_pinned",
            "benchmark_semantics_version": BENCHMARK_SEMANTICS_VERSION,
        }
        status = (
            Status.PASS
            if all(torch.isfinite(torch.tensor(losses))) and finite_gradients
            else Status.FAIL
        )
    caps_after = collect_capabilities()
    metrics["telemetry_before"] = caps_before.get("nvidia_smi")
    metrics["telemetry_after"] = caps_after.get("nvidia_smi")
    metrics["telemetry_during"] = "not sampled; CUDA timing region kept exclusive"
    with RunWriter(root, "single") as writer:
        writer.json("environment.json", caps_after)
        writer.json(
            "hardware.json",
            {"before": caps_before["gpu"], "after": caps_after["gpu"]},
        )
        writer.json(
            "topology.json",
            {"before": caps_before.get("topology"), "after": caps_after.get("topology")},
        )
        writer.json(
            "software_versions.json",
            {"torch": torch.__version__, "cuda": torch.version.cuda},
        )
        writer.json("dataset_manifest.json", dataset_manifest)
        writer.json(
            "sample_schedule.json",
            {
                "seed": schedule.seed,
                "updates": schedule.updates,
                "global_batch": schedule.global_batch,
                "sha256": schedule.fingerprint(),
                "mapping": "sample_id modulo persisted dataset row count",
                "mapped_row_ids": mapped_rows,
            },
        )
        writer.json("model_config.json", asdict(config))
        writer.json(
            "strategy_config.json",
            {
                "strategy": "single",
                "torch_compile": False,
                "global_batch": 2,
                "input_mode": "host_pinned",
                "h2d_excluded_from_model_update_timing": True,
                "benchmark_semantics_version": BENCHMARK_SEMANTICS_VERSION,
            },
        )
        writer.json(
            "initial_state_fingerprint.json",
            {"sha256": state_fingerprint(canonical_model(config)), "seed": 17},
        )
        writer.json(
            "correctness.json",
            {"finite_loss": status is Status.PASS, "finite_gradients": finite_gradients},
        )
        writer.json("metrics.json", metrics)
        pd.DataFrame(
            {"update": list(range(measured)), "update_time_s": times, "loss": losses}
        ).to_parquet(writer.path / "metrics.parquet", index=False)
        pd.DataFrame(
            {"rank": [0], "update_time_p50_s": [metrics.get("step_p50_s", 0.0)]}
        ).to_parquet(writer.path / "rank_metrics.parquet", index=False)
        pd.DataFrame(columns=["collective", "duration_s"]).to_parquet(
            writer.path / "collective_metrics.parquet", index=False
        )
        writer.json("planner_decision.json", {"applicable": False})
        writer.json("fault_events.json", [])
        writer.json("recovery.json", {"applicable": False})
        writer.finalize({"status": status, "performance_claim": status is Status.PASS})
        return status, writer.path, metrics
