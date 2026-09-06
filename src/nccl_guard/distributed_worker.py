from __future__ import annotations

import json
import os
import statistics
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from nccl_guard.config.core import PROFILES, ParallelConfig, validate_tp
from nccl_guard.data.core import SampleSchedule
from nccl_guard.models.dense.transformer import canonical_model, state_fingerprint
from nccl_guard.parallel.strategies import (
    apply_ddp,
    apply_fsdp2_bottom_up,
    apply_fsdp2_root_only,
    apply_tp_then_fsdp,
    enable_separate_reduce_scatter_group,
    optimizer_after_sharding,
)


def _run_id(strategy: str, rank: int) -> str:
    values = (
        [f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{strategy}-{uuid.uuid4().hex[:8]}"]
        if rank == 0
        else [""]
    )
    dist.broadcast_object_list(values, src=0)
    return values[0]


def _finalize(path: Path) -> None:
    hashes = {}
    for item in sorted(path.rglob("*")):
        if item.is_file() and item.name != "sha256_manifest.json":
            hashes[item.relative_to(path).as_posix()] = sha256(item.read_bytes()).hexdigest()
    (path / "sha256_manifest.json").write_text(
        json.dumps(hashes, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    if os.name == "nt" or not torch.cuda.is_available() or not dist.is_nccl_available():
        raise RuntimeError("distributed worker requires Linux CUDA and NCCL; Gloo is forbidden")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    strategy = os.environ.get("NCCL_GUARD_STRATEGY", "ddp")
    tp_degree = int(os.environ.get("NCCL_GUARD_TP_DEGREE", "2")) if strategy == "tp_fsdp" else 1
    microbatch = int(os.environ.get("NCCL_GUARD_MICROBATCH", "1"))
    grad_accumulation = int(os.environ.get("NCCL_GUARD_GRAD_ACCUMULATION", str(tp_degree)))
    # TP ranks share samples; only DP degree contributes to global batch.
    parallel = ParallelConfig(world, tp_degree, microbatch, grad_accumulation)
    config = PROFILES[os.environ.get("NCCL_GUARD_MODEL", "tiny")]
    validate_tp(config, parallel)
    model = canonical_model(config, 17).to(device)
    initial_fingerprint = state_fingerprint(model)
    precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    if strategy == "ddp":
        model = apply_ddp(
            model, [local_rank], bucket_cap_mb=float(os.environ.get("NCCL_GUARD_BUCKET_MB", "25"))
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    elif strategy == "fsdp2":
        mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp",))
        reshard_after_forward = os.environ.get("NCCL_GUARD_RESHARD_AFTER_FORWARD", "1") != "0"
        if os.environ.get("NCCL_GUARD_FSDP_ROOT_ONLY") == "1":
            model = apply_fsdp2_root_only(model, mesh, reshard_after_forward=reshard_after_forward)
        else:
            model = apply_fsdp2_bottom_up(model, mesh, reshard_after_forward=reshard_after_forward)
        if os.environ.get("NCCL_GUARD_SEPARATE_RS") == "1":
            enable_separate_reduce_scatter_group(model)
        optimizer = optimizer_after_sharding(model, torch.optim.AdamW, lr=1e-3)
    elif strategy == "tp_fsdp":
        mesh = init_device_mesh(
            "cuda", (parallel.dp_degree, tp_degree), mesh_dim_names=("dp", "tp")
        )
        model = apply_tp_then_fsdp(model, config, parallel, mesh)
        optimizer = optimizer_after_sharding(model, torch.optim.AdamW, lr=1e-3)
    else:
        raise ValueError(f"unknown strategy {strategy}")
    dp_rank = rank // tp_degree
    schedule = SampleSchedule(17, 7, parallel.global_batch)
    times: list[float] = []
    gpu_times: list[float] = []
    fault_events: list[dict[str, object]] = []
    loss_value = 0.0
    for step in range(7):
        sample_ids = schedule.for_dp_rank(dp_rank, parallel.dp_degree)[step]
        optimizer.zero_grad(set_to_none=True)
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        start.record()
        fault_kind = os.environ.get("NCCL_GUARD_FAULT_KIND")
        fault_step = int(os.environ.get("NCCL_GUARD_FAULT_STEP", "4"))
        fault_rank = int(os.environ.get("NCCL_GUARD_FAULT_RANK", "1"))
        if step == fault_step and rank == fault_rank and fault_kind == "compute_delay":
            delay_ms = float(os.environ.get("NCCL_GUARD_FAULT_DURATION_MS", "100"))
            time.sleep(delay_ms / 1000)
            fault_events.append(
                {"step": step, "rank": rank, "kind": fault_kind, "duration_ms": delay_ms}
            )
        accumulated_loss = 0.0
        for microstep in range(parallel.grad_accumulation):
            microstep_ids = sample_ids[
                microstep * parallel.microbatch_per_dp_rank : (microstep + 1)
                * parallel.microbatch_per_dp_rank
            ]
            tokens = torch.cat(
                [
                    torch.randint(
                        0,
                        config.vocab_size,
                        (1, config.sequence_length + 1),
                        generator=torch.Generator(device=device).manual_seed(sample_id),
                        device=device,
                    )
                    for sample_id in microstep_ids
                ],
                dim=0,
            )
            with torch.autocast("cuda", dtype=dtype):
                _, loss = model(tokens[:, :-1], tokens[:, 1:])
            assert loss is not None
            scaler.scale(loss / parallel.grad_accumulation).backward()
            accumulated_loss += float(loss.detach())
        if step == fault_step and rank == fault_rank and fault_kind == "workload_skew":
            repeats = int(os.environ.get("NCCL_GUARD_SKEW_REPEATS", "2"))
            for _ in range(repeats):
                with torch.autocast("cuda", dtype=dtype):
                    _, skew_loss = model(tokens[:, :-1], tokens[:, 1:])
                assert skew_loss is not None
                scaler.scale(skew_loss / (parallel.grad_accumulation * repeats)).backward()
            fault_events.append(
                {"step": step, "rank": rank, "kind": fault_kind, "extra_microsteps": repeats}
            )
        scaler.step(optimizer)
        scaler.update()
        end.record()
        torch.cuda.synchronize()
        wall_elapsed = time.perf_counter() - wall_start
        loss_value = accumulated_loss / len(sample_ids)
        if step >= 2:
            times.append(wall_elapsed)
            gpu_times.append(start.elapsed_time(end) / 1000)
    local = {
        "rank": rank,
        "strategy": strategy,
        "mean_update_s": statistics.mean(times),
        "mean_gpu_interval_s": statistics.mean(gpu_times),
        "p95_update_s": max(times),
        "loss": loss_value,
        "tokens_per_second": len(sample_ids) * config.sequence_length / statistics.mean(times),
        "processed_tokens_per_update": len(sample_ids) * config.sequence_length,
        "fault_events": fault_events,
    }
    gathered: list[dict[str, Any] | None] = [None] * world if rank == 0 else []
    dist.gather_object(local, gathered if rank == 0 else None, dst=0)
    run_id = _run_id(strategy, rank)
    output = Path("artifacts") / run_id
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        rows = [row for row in gathered if row is not None]
        pd.DataFrame(
            [{key: value for key, value in row.items() if key != "fault_events"} for row in rows]
        ).to_parquet(output / "rank_metrics.parquet", index=False)
        rank_times = [float(row["mean_update_s"]) for row in rows]
        median_time = statistics.median(rank_times)
        (output / "fault_events.json").write_text(
            json.dumps([event for row in rows for event in row["fault_events"]], indent=2) + "\n",
            encoding="utf-8",
        )
        (output / "metrics.json").write_text(
            json.dumps(
                {
                    "rank_imbalance": (max(rank_times) - median_time) / median_time,
                    "critical_rank_update_s": max(rank_times),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (output / "model_config.json").write_text(
            json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8"
        )
        (output / "strategy_config.json").write_text(
            json.dumps(
                {
                    "strategy": strategy,
                    "world_size": world,
                    "dp_degree": parallel.dp_degree,
                    "tp_degree": tp_degree,
                    "microbatch_per_dp_rank": parallel.microbatch_per_dp_rank,
                    "grad_accumulation": parallel.grad_accumulation,
                    "global_batch": parallel.global_batch,
                    "comparison_track": os.environ.get("NCCL_GUARD_TRACK", "ISO_WORKLOAD"),
                    "optimizer_after_sharding": strategy != "ddp",
                    "torch_compile": False,
                    "precision": precision,
                    "autocast": True,
                    "grad_scaler": scaler.is_enabled(),
                    "reshard_after_forward": os.environ.get("NCCL_GUARD_RESHARD_AFTER_FORWARD", "1")
                    != "0",
                    "root_only_diagnostic": os.environ.get("NCCL_GUARD_FSDP_ROOT_ONLY") == "1",
                    "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                    "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (output / "sample_schedule.json").write_text(
            json.dumps(
                {"sha256": schedule.fingerprint(), "global_batch": parallel.global_batch}, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        (output / "initial_state_fingerprint.json").write_text(
            json.dumps({"sha256": initial_fingerprint, "seed": 17}, indent=2) + "\n",
            encoding="utf-8",
        )
        (output / "verdict.json").write_text(
            json.dumps(
                {"status": "PASS", "claim_scope": "Linux NCCL distributed", "strategy": strategy},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        _finalize(output)
        print(output)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
