from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

SCIENTIFIC_FILES = [
    "src/nccl_guard/tuning/protocol.py",
    "src/nccl_guard/tuning/simulation.py",
    "src/nccl_guard/diagnosis/tracelens.py",
    "src/nccl_guard/config/core.py",
    "src/nccl_guard/metrics/core.py",
    "src/nccl_guard/parallel/strategies.py",
    "src/nccl_guard/distributed_worker.py",
    "src/nccl_guard/recovery/state.py",
    "src/nccl_guard/recovery_worker.py",
    "src/nccl_guard/experiments/runner.py",
    "configs/models/tiny.yaml",
    "configs/models/small.yaml",
    "configs/models/medium.yaml",
    "configs/models/large.yaml",
    "configs/qualification/local-gpu.yaml",
    "configs/experiments/e00.yaml",
    "configs/experiments/e01.yaml",
    "configs/experiments/e02.yaml",
    "configs/experiments/e03.yaml",
    "configs/experiments/e04.yaml",
    "configs/experiments/e05.yaml",
    "configs/experiments/e06.yaml",
    "configs/experiments/e07.yaml",
    "configs/experiments/e08.yaml",
    "configs/experiments/e09.yaml",
    "configs/experiments/e10.yaml",
    "configs/experiments/e11.yaml",
    "configs/experiments/e12.yaml",
    "configs/hardware/local-windows.yaml",
    "configs/datasets/bounded.yaml",
    "configs/strategies/v1.yaml",
    "data/manifests/synthetic_dense.json",
    "data/manifests/wikitext103.json",
    "data/manifests/c4.json",
    "data/manifests/criteo.json",
]


def freeze_planner(root: Path) -> Path:
    files = {name: sha256((root / name).read_bytes()).hexdigest() for name in SCIENTIFIC_FILES}
    path = root / "evidence/planner_freeze_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text(encoding="utf-8")).get("transaction_state") in {
        "STARTED",
        "COMPLETED",
    }:
        raise RuntimeError("final confirmation already started; v1 freeze is immutable")
    git_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    commit = git_result.stdout.strip() if git_result.returncode == 0 else None
    value = {
        "version": "v1",
        "created_at": datetime.now(UTC).isoformat(),
        "transaction_state": "CREATED",
        "candidate_budget": 3,
        "git_commit": commit,
        "files": files,
    }
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path
