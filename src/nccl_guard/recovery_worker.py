from __future__ import annotations

import io
import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from nccl_guard.config.core import ModelConfig
from nccl_guard.data.core import SampleSchedule
from nccl_guard.models.dense.transformer import canonical_model, state_fingerprint
from nccl_guard.parallel.strategies import apply_fsdp2_bottom_up, optimizer_after_sharding
from nccl_guard.recovery.state import ApplicationState, dcp_load, dcp_save


def _write_rank(path: Path, rank: int, value: dict[str, Any]) -> None:
    rank_path = path / f"rank-{rank}.json"
    rank_path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _fingerprint(value: Any) -> str:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return sha256(buffer.getvalue()).hexdigest()


def main() -> int:
    if os.name == "nt" or not torch.cuda.is_available() or not dist.is_nccl_available():
        raise RuntimeError("elastic recovery qualification requires Linux CUDA and NCCL")
    local_rank = int(os.environ["LOCAL_RANK"])
    restart_count = int(os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    output = Path(os.environ["NCCL_GUARD_RECOVERY_OUTPUT"])
    checkpoint = os.environ["NCCL_GUARD_RECOVERY_CHECKPOINT"]
    output.mkdir(parents=True, exist_ok=True)

    config = ModelConfig("recovery", 2, 128, 4, 512, 8192, 128)
    model = canonical_model(config, 17).to(device)
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp",))
    model = apply_fsdp2_bottom_up(model, mesh)
    optimizer = optimizer_after_sharding(model, torch.optim.AdamW, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    schedule = SampleSchedule(41, 6, world)
    app = ApplicationState.capture(0, 0, schedule.fingerprint(), config.fingerprint(), scheduler)
    if restart_count:
        dcp_load(
            checkpoint,
            model,
            optimizer,
            app,
            scheduler,
            expected_dataset_fingerprint=schedule.fingerprint(),
            expected_config_fingerprint=config.fingerprint(),
        )

    losses: list[float] = []
    consumed = list(app.consumed_sample_ids)
    for step in range(app.global_step, schedule.updates):
        if (
            os.environ.get("NCCL_GUARD_INJECT_FAILURE") == "1"
            and restart_count == 0
            and rank == 1
            and step == 3
        ):
            os._exit(17)
        sample_id = schedule.for_dp_rank(rank, world)[step][0]
        generator = torch.Generator(device=device).manual_seed(sample_id)
        tokens = torch.randint(
            0,
            config.vocab_size,
            (1, config.sequence_length + 1),
            generator=generator,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(tokens[:, :-1], tokens[:, 1:])
        assert loss is not None
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach()))
        consumed.append(sample_id)
        app = ApplicationState.capture(
            step + 1,
            (step + 1) * world,
            schedule.fingerprint(),
            config.fingerprint(),
            scheduler,
            consumed,
        )
        if step + 1 == 3:
            dcp_save(checkpoint, model, optimizer, app)
            dist.barrier()

    expected = [row[0] for row in schedule.for_dp_rank(rank, world)]
    local = {
        "rank": rank,
        "restart_count": restart_count,
        "global_step": app.global_step,
        "data_cursor": app.data_cursor,
        "consumed_sample_ids": consumed,
        "no_duplicate_or_skip": consumed == expected,
        "finite_loss": all(torch.isfinite(torch.tensor(losses))),
        "model_fingerprint": state_fingerprint(model),
        "optimizer_fingerprint": _fingerprint(optimizer.state_dict()),
        "scheduler_fingerprint": _fingerprint(scheduler.state_dict()),
        "rng_fingerprint": _fingerprint(
            {
                "python": app.python_rng,
                "numpy": app.numpy_rng,
                "torch": app.torch_rng,
                "cuda": app.cuda_rng,
            }
        ),
        "last_loss": losses[-1],
    }
    _write_rank(output, rank, local)
    dist.barrier()
    if rank == 0:
        rows = [json.loads((output / f"rank-{item}.json").read_text()) for item in range(world)]
        verdict = {
            "status": "PASS"
            if all(row["no_duplicate_or_skip"] and row["finite_loss"] for row in rows)
            else "FAIL",
            "same_world_size_recovery": True,
            "world_size_changing_recovery": "NOT_VERIFIED",
            "ranks": rows,
        }
        (output / "recovery.json").write_text(
            json.dumps(verdict, indent=2) + "\n", encoding="utf-8"
        )
        hashes = {
            item.name: sha256(item.read_bytes()).hexdigest()
            for item in sorted(output.iterdir())
            if item.is_file() and item.name != "sha256_manifest.json"
        }
        (output / "sha256_manifest.json").write_text(
            json.dumps(hashes, indent=2) + "\n", encoding="utf-8"
        )
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
