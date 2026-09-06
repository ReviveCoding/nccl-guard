from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SparseConfig:
    dense_features: int = 13
    categorical_features: int = 26
    buckets: int = 100_003
    embedding_dim: int = 64


class SyntheticSparseModel(nn.Module):
    """Schema-compatible local control model; never reported as Criteo evidence."""

    def __init__(self, config: SparseConfig) -> None:
        super().__init__()
        self.config = config
        self.dense = nn.Sequential(
            nn.Linear(config.dense_features, 128), nn.ReLU(), nn.Linear(128, 64)
        )
        self.embeddings = nn.ModuleList(
            [
                nn.EmbeddingBag(config.buckets, config.embedding_dim, mode="mean")
                for _ in range(config.categorical_features)
            ]
        )
        self.over = nn.Sequential(
            nn.Linear(64 + config.categorical_features * config.embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, dense: Tensor, values: list[Tensor], offsets: list[Tensor]) -> Tensor:
        embedded = [
            table(value, offset)
            for table, value, offset in zip(self.embeddings, values, offsets, strict=True)
        ]
        return self.over(torch.cat([self.dense(dense), *embedded], dim=-1)).squeeze(-1)


def torchrec_capability() -> tuple[bool, str]:
    try:
        import fbgemm_gpu  # noqa: F401
        import torchrec  # noqa: F401
    except Exception as exc:
        return False, f"TorchRec/FBGEMM unavailable: {exc!r}"
    return True, "TorchRec and FBGEMM imported; execution capability still requires CUDA preflight"


def torchrec_sharding_plan(
    model: Any, topology: Any, sharders: list[Any], process_group: Any
) -> Any:
    """Run the official TorchRec collective planner as the S3 baseline."""
    from torchrec.distributed.planner import EmbeddingShardingPlanner

    planner = EmbeddingShardingPlanner(topology=topology)
    return planner.collective_plan(model, sharders, process_group)


def build_torchrec_embedding_collection(config: SparseConfig) -> Any:
    """Build the EBC used by S0-S3; importing is capability-gated."""
    from torchrec.modules.embedding_configs import EmbeddingBagConfig
    from torchrec.modules.embedding_modules import EmbeddingBagCollection

    tables = [
        EmbeddingBagConfig(
            name=f"table_{index}",
            embedding_dim=config.embedding_dim,
            num_embeddings=config.buckets,
            feature_names=[f"categorical_{index}"],
        )
        for index in range(config.categorical_features)
    ]
    return EmbeddingBagCollection(tables=tables)
