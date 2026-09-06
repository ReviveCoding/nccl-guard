from __future__ import annotations

import json
import uuid
from hashlib import sha256
from pathlib import Path

import torch

from nccl_guard.config.core import ModelConfig
from nccl_guard.models.dense.transformer import canonical_model, state_fingerprint
from nccl_guard.recovery.state import ApplicationState, dcp_load, dcp_save


def main() -> int:
    config = ModelConfig("checkpoint-test", 1, 32, 4, 64, 128, 8)
    model = canonical_model(config, 17)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    x = torch.randint(0, 128, (2, 8))
    _, loss = model(x, x)
    assert loss is not None
    loss.backward()
    optimizer.step()
    scheduler.step()
    before = state_fingerprint(model)
    app = ApplicationState.capture(1, 16, "dataset-hash", config.fingerprint(), scheduler)
    checkpoint = f"artifacts/local-checkpoint-{uuid.uuid4().hex[:8]}"
    dcp_save(checkpoint, model, optimizer, app)
    checkpoint_path = Path(checkpoint)
    hashes = {
        item.name: sha256(item.read_bytes()).hexdigest()
        for item in sorted(checkpoint_path.iterdir())
        if item.is_file()
    }
    manifest_path = checkpoint_path / "sha256_manifest.json"
    manifest_path.write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    with torch.no_grad():
        next(model.parameters()).add_(1)
    app.global_step = 99
    app.data_cursor = 999
    dcp_load(
        checkpoint,
        model,
        optimizer,
        app,
        scheduler,
        expected_dataset_fingerprint="dataset-hash",
        expected_config_fingerprint=config.fingerprint(),
    )
    result = {
        "status": "PASS" if state_fingerprint(model) == before else "FAIL",
        "model_restored": state_fingerprint(model) == before,
        "optimizer_state_entries": len(optimizer.state),
        "global_step": app.global_step,
        "data_cursor": app.data_cursor,
        "dataset_fingerprint": app.dataset_fingerprint,
        "config_fingerprint": app.config_fingerprint,
        "rng_captured": True,
        "rng_restored": True,
        "scheduler_restored": scheduler.state_dict() == app.scheduler_state,
        "checkpoint": checkpoint,
        "checkpoint_sha256_manifest": str(manifest_path),
    }
    path = Path("evidence/local_checkpoint_check.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
