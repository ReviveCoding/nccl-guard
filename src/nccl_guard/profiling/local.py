from __future__ import annotations

import json
import uuid
from pathlib import Path

import torch

from nccl_guard.config.core import PROFILES
from nccl_guard.models.dense.transformer import canonical_model
from nccl_guard.profiling.core import phase, profile_cuda_step


def main() -> int:
    if not torch.cuda.is_available():
        print(json.dumps({"status": "FAIL_GPU_EXECUTION"}))
        return 1
    model = canonical_model(PROFILES["local"]).cuda()
    optimizer = torch.optim.AdamW(model.parameters())
    tokens = torch.randint(0, PROFILES["local"].vocab_size, (2, 129), device="cuda")

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with phase("forward"):
            _, loss = model(tokens[:, :-1], tokens[:, 1:])
        assert loss is not None
        with phase("backward"):
            loss.backward()
        with phase("optimizer"):
            optimizer.step()

    profile_path = Path("artifacts") / f"local-profiler-{uuid.uuid4().hex[:8]}"
    result = profile_cuda_step(profile_path, step)
    summary = Path("evidence/local_profiler.json")
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
