from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch


@contextmanager
def phase(name: str) -> Iterator[None]:
    with torch.profiler.record_function(f"nccl_guard::{name}"):
        yield


def profile_cuda_step(path: Path, step: Any) -> dict[str, object]:
    path.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities, record_shapes=True, profile_memory=True
    ) as p:
        step()
        torch.cuda.synchronize()
    trace = path / "trace.json"
    p.export_chrome_trace(str(trace))
    manifest = path / "sha256_manifest.json"
    manifest.write_text(
        json.dumps({trace.name: sha256(trace.read_bytes()).hexdigest()}, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "PASS",
        "ecr_kind": "ECR_APPROX",
        "trace": str(trace),
        "sha256_manifest": str(manifest),
    }
