from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset


def main() -> int:
    if not torch.cuda.is_available():
        return 1
    values = torch.randint(0, 8192, (512, 129))
    dataset = TensorDataset(values)
    results: list[dict[str, Any]] = []
    for workers in [0, 2, 4, 8]:
        loader = (
            DataLoader(
                dataset,
                batch_size=8,
                num_workers=workers,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
            )
            if workers
            else DataLoader(dataset, batch_size=8, num_workers=0, pin_memory=True)
        )
        start = time.perf_counter()
        batches = 0
        try:
            for (batch,) in loader:
                _ = batch.cuda(non_blocking=True)
                batches += 1
                if batches == 10:
                    break
            torch.cuda.synchronize()
            results.append(
                {
                    "workers": workers,
                    "seconds": time.perf_counter() - start,
                    "batches": batches,
                    "pin_memory": True,
                    "non_blocking": True,
                    "status": "PASS",
                }
            )
        except (OSError, PermissionError) as exc:
            results.append({"workers": workers, "status": "SKIP_UNSUPPORTED", "reason": repr(exc)})
    passed = [row for row in results if row["status"] == "PASS"]
    best = min(passed, key=lambda row: float(row["seconds"]))
    path = Path("evidence/dataloader_preflight.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "results": results,
                "selected_workers": best["workers"],
                "prefetch_factor": 2,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
