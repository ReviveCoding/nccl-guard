from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import psutil
import torch


def _run(command: list[str]) -> str | None:
    try:
        return (
            subprocess.run(
                command, capture_output=True, text=True, timeout=15, check=False
            ).stdout.strip()
            or None
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def collect_capabilities() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    gpu: dict[str, Any] = {"count": torch.cuda.device_count(), "available": cuda}
    if cuda:
        props = torch.cuda.get_device_properties(0)
        gpu.update(
            name=props.name,
            total_vram=props.total_memory,
            compute_capability=[props.major, props.minor],
        )
    smi = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version,power.limit,power.draw,temperature.gpu,clocks.sm,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    gpu_telemetry: list[dict[str, Any]] = []
    if smi:
        fields = [
            "name",
            "memory_total_mib",
            "memory_free_mib",
            "driver_version",
            "power_limit_w",
            "power_draw_w",
            "temperature_c",
            "sm_clock_mhz",
            "utilization_percent",
        ]
        for line in smi.splitlines():
            values = [value.strip() for value in line.split(",")]
            gpu_telemetry.append(dict(zip(fields, values, strict=False)))
    nccl_version: object = None
    if torch.distributed.is_available() and torch.distributed.is_nccl_available():
        try:
            nccl_version = torch.cuda.nccl.version()
        except Exception as exc:
            nccl_version = {"query_error": repr(exc)}
    result: dict[str, Any] = {
        "os": platform.platform(),
        "python": sys.version,
        "cpu": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
        "powershell_version": _run(
            ["pwsh", "-NoProfile", "-Command", "$PSVersionTable.PSVersion.ToString()"]
        ),
        "gpu": gpu,
        "nvidia_smi": smi,
        "gpu_telemetry": gpu_telemetry,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "bf16": bool(cuda and torch.cuda.is_bf16_supported()),
        "distributed": {
            "available": torch.distributed.is_available(),
            "nccl": bool(
                torch.distributed.is_available() and torch.distributed.is_nccl_available()
            ),
            "gloo": bool(
                torch.distributed.is_available() and torch.distributed.is_gloo_available()
            ),
            "nccl_version": nccl_version,
        },
        "tools": {
            name: shutil.which(name)
            for name in ["nsys", "mpiexec", "ibstat", "rdma", "all_reduce_perf"]
        },
        "topology": _run(["nvidia-smi", "topo", "-m"]),
        "native_windows": os.name == "nt",
        "network_interfaces": sorted(psutil.net_if_addrs()),
    }
    for package in ["torchrec", "fbgemm_gpu", "torchcomms", "pynvml"]:
        try:
            __import__(package)
            result.setdefault("packages", {})[package] = {
                "importable": True,
                "execution_verified": False,
            }
        except Exception:
            result.setdefault("packages", {})[package] = {
                "importable": False,
                "execution_verified": False,
            }
    return result


def write_capabilities(root: Path) -> Path:
    path = root / "evidence" / "hardware_capabilities.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(collect_capabilities(), indent=2, default=str) + "\n", encoding="utf-8"
    )
    return path
