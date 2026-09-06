from __future__ import annotations

import math
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from nccl_guard.data.core import SampleSchedule
from nccl_guard.evidence.writer import RunWriter
from nccl_guard.hardware.doctor import collect_capabilities
from nccl_guard.models.sparse.model import SparseConfig, SyntheticSparseModel
from nccl_guard.status import Status


def run_sparse_cuda(
    root: Path, *, warmup: int = 5, measured: int = 20, batch_size: int = 128
) -> tuple[Status, Path, dict[str, Any]]:
    """Run the schema-compatible synthetic sparse control workload on CUDA only."""
    if not torch.cuda.is_available():
        with RunWriter(root, "sparse") as writer:
            writer.finalize(
                {"status": Status.FAIL_GPU_EXECUTION, "reason": "CUDA required; no CPU fallback"}
            )
            return Status.FAIL_GPU_EXECUTION, writer.path, {}

    config = SparseConfig(buckets=4096, embedding_dim=32)
    torch.manual_seed(29)
    model = SyntheticSparseModel(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    generator = torch.Generator(device="cuda").manual_seed(29)
    total_updates = warmup + measured
    schedule = SampleSchedule(29, total_updates, batch_size)
    dense = torch.randn(
        total_updates,
        batch_size,
        config.dense_features,
        generator=generator,
        device="cuda",
    )
    values = [
        torch.randint(
            0,
            config.buckets,
            (total_updates, batch_size),
            generator=generator,
            device="cuda",
        )
        for _ in range(config.categorical_features)
    ]
    offsets = [torch.arange(batch_size, device="cuda") for _ in values]
    target = torch.randint(
        0, 2, (total_updates, batch_size), generator=generator, device="cuda"
    ).float()

    times: list[float] = []
    losses: list[float] = []
    metrics: dict[str, Any]
    finite_gradients = True
    torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(warmup + measured):
            optimizer.zero_grad(set_to_none=True)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            with torch.autocast("cuda", dtype=dtype):
                logits = model(dense[step], [value[step] for value in values], offsets)
                loss = F.binary_cross_entropy_with_logits(logits.float(), target[step])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            end.record()
            torch.cuda.synchronize()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            finite_gradients = finite_gradients and all(
                bool(torch.isfinite(gradient).all()) for gradient in gradients
            )
            if step >= warmup:
                times.append(start.elapsed_time(end) / 1000)
                losses.append(float(loss.detach()))
    except torch.cuda.OutOfMemoryError as exc:
        status, metrics = Status.VALID_BUT_OOM, {"error": str(exc)}
    except Exception as exc:
        status, metrics = Status.FAIL_GPU_EXECUTION, {"error": repr(exc)}
    else:
        ordered = sorted(times)
        mean_time = statistics.mean(times)
        metrics = {
            "samples_per_second": batch_size / mean_time,
            "update_time_mean_s": mean_time,
            "step_p50_s": statistics.median(times),
            "step_p95_s": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "loss": losses[-1],
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "precision": precision,
            "autocast": True,
            "grad_scaler": scaler.is_enabled(),
            "device": str(next(model.parameters()).device),
            "warmup_updates": warmup,
            "measured_updates": measured,
            "input_mode": "gpu_resident",
        }
        status = (
            Status.PASS if finite_gradients and all(map(math.isfinite, losses)) else Status.FAIL
        )

    capabilities = collect_capabilities()
    with RunWriter(root, "sparse") as writer:
        writer.json("environment.json", capabilities)
        writer.json("hardware.json", capabilities["gpu"])
        writer.json("topology.json", capabilities.get("topology"))
        writer.json(
            "software_versions.json", {"torch": torch.__version__, "cuda": torch.version.cuda}
        )
        writer.json(
            "dataset_manifest.json",
            {
                "name": "synthetic-sparse-schema-compatible",
                "source": "deterministic generator",
                "license": "CC0",
                "seed": 29,
                "not_criteo_evidence": True,
            },
        )
        writer.json(
            "sample_schedule.json",
            {
                "seed": schedule.seed,
                "global_batch": schedule.global_batch,
                "updates": schedule.updates,
                "sha256": schedule.fingerprint(),
            },
        )
        writer.json("model_config.json", asdict(config))
        writer.json(
            "strategy_config.json",
            {"strategy": "synthetic_sparse_unsharded", "input_mode": "gpu_resident"},
        )
        writer.json("initial_state_fingerprint.json", {"seed": 29})
        writer.json(
            "correctness.json",
            {"finite_loss": status is Status.PASS, "finite_gradients": finite_gradients},
        )
        writer.json("metrics.json", metrics)
        pd.DataFrame(
            {"update": range(len(times)), "update_time_s": times, "loss": losses}
        ).to_parquet(writer.path / "metrics.parquet", index=False)
        pd.DataFrame({"rank": [0]}).to_parquet(writer.path / "rank_metrics.parquet", index=False)
        pd.DataFrame(columns=["collective", "duration_s"]).to_parquet(
            writer.path / "collective_metrics.parquet", index=False
        )
        writer.json("planner_decision.json", {"applicable": False})
        writer.json("fault_events.json", [])
        writer.json("recovery.json", {"applicable": False})
        writer.finalize(
            {
                "status": status,
                "performance_claim": status is Status.PASS,
                "claim_scope": "synthetic sparse single-GPU CUDA only",
            }
        )
        return status, writer.path, metrics
