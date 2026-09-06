from __future__ import annotations

from typing import Any

from nccl_guard.config.core import ModelConfig, ParallelConfig, validate_tp

DDP_BUCKET_CANDIDATES_MB = (12.5, 25.0, 50.0, 100.0)


def apply_ddp(
    model: Any,
    device_ids: list[int],
    *,
    bucket_cap_mb: float = 25.0,
    static_graph: bool = False,
) -> Any:
    from torch.nn.parallel import DistributedDataParallel

    return DistributedDataParallel(
        model,
        device_ids=device_ids,
        gradient_as_bucket_view=True,
        find_unused_parameters=False,
        bucket_cap_mb=bucket_cap_mb,
        static_graph=static_graph,
    )


def apply_fsdp2_bottom_up(model: Any, mesh: Any, *, reshard_after_forward: bool = True) -> Any:
    from torch.distributed.fsdp import fully_shard

    for block in model.blocks:
        fully_shard(block, mesh=mesh, reshard_after_forward=reshard_after_forward)
    fully_shard(model, mesh=mesh, reshard_after_forward=reshard_after_forward)
    model._nccl_guard_sharded = True
    return model


def apply_fsdp2_root_only(model: Any, mesh: Any, *, reshard_after_forward: bool = True) -> Any:
    """F0 diagnostic only; layer-wise bottom-up remains the strong FSDP2 baseline."""
    from torch.distributed.fsdp import fully_shard

    fully_shard(model, mesh=mesh, reshard_after_forward=reshard_after_forward)
    model._nccl_guard_sharded = True
    return model


def apply_tp_then_fsdp(model: Any, config: ModelConfig, parallel: ParallelConfig, mesh: Any) -> Any:
    validate_tp(config, parallel)
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        SequenceParallel,
        parallelize_module,
    )

    tp_mesh = mesh["tp"]
    plan: dict[str, Any] = {"token_embedding": RowwiseParallel(), "final_norm": SequenceParallel()}
    for index in range(config.layers):
        prefix = f"blocks.{index}"
        plan.update(
            {
                f"{prefix}.attn_norm": SequenceParallel(),
                f"{prefix}.mlp_norm": SequenceParallel(),
                f"{prefix}.attn.q_proj": ColwiseParallel(),
                f"{prefix}.attn.k_proj": ColwiseParallel(),
                f"{prefix}.attn.v_proj": ColwiseParallel(),
                f"{prefix}.attn.out_proj": RowwiseParallel(),
                f"{prefix}.mlp.gate_proj": ColwiseParallel(),
                f"{prefix}.mlp.up_proj": ColwiseParallel(),
                f"{prefix}.mlp.down_proj": RowwiseParallel(),
            }
        )
    parallelize_module(model, tp_mesh, plan)
    model._nccl_guard_tp_applied = True
    return apply_fsdp2_bottom_up(model, mesh["dp"])


def optimizer_after_sharding(model: Any, optimizer_type: Any, **kwargs: Any) -> Any:
    if not getattr(model, "_nccl_guard_sharded", False):
        raise RuntimeError("optimizer must be constructed after FSDP2 sharding")
    return optimizer_type(model.parameters(), **kwargs)


def enable_separate_reduce_scatter_group(model: Any) -> None:
    method = getattr(model, "set_separate_reduce_scatter_group", None)
    if method is None:
        raise NotImplementedError("installed FSDP2 lacks separate ReduceScatter group support")
    method(True, recurse=True)
