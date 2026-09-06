from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from hashlib import sha256


@dataclass(frozen=True)
class ModelConfig:
    name: str
    layers: int
    hidden: int
    heads: int
    ffn: int
    vocab_size: int = 50304
    sequence_length: int = 512

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    def validate(self) -> None:
        if min(self.layers, self.hidden, self.heads, self.ffn, self.vocab_size) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.hidden % self.heads:
            raise ValueError("hidden must be divisible by heads")

    def fingerprint(self) -> str:
        return sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


PROFILES = {
    "tiny": ModelConfig("tiny", 4, 256, 4, 1024),
    "small": ModelConfig("small", 12, 768, 12, 3072),
    "medium": ModelConfig("medium", 24, 1024, 16, 4096),
    "large": ModelConfig("large", 32, 1536, 24, 6144),
    # Deliberate hardware-adaptive characterization profile, not a renamed headline profile.
    "local": ModelConfig("local", 2, 128, 4, 512, vocab_size=8192, sequence_length=128),
}


@dataclass(frozen=True)
class ParallelConfig:
    world_size: int
    tp_degree: int = 1
    microbatch_per_dp_rank: int = 1
    grad_accumulation: int = 1
    scaling: str = "strong"

    @property
    def dp_degree(self) -> int:
        if self.tp_degree <= 0 or self.world_size % self.tp_degree:
            raise ValueError("world_size must be divisible by tp_degree")
        return self.world_size // self.tp_degree

    @property
    def global_batch(self) -> int:
        return self.microbatch_per_dp_rank * self.grad_accumulation * self.dp_degree

    def global_tokens(self, sequence_length: int) -> int:
        return self.global_batch * sequence_length


def validate_tp(model: ModelConfig, parallel: ParallelConfig) -> None:
    _ = parallel.dp_degree
    for label, value in {
        "hidden": model.hidden,
        "heads": model.heads,
        "ffn": model.ffn,
        "vocab": model.vocab_size,
    }.items():
        if value % parallel.tp_degree:
            raise ValueError(f"{label} is not divisible by tp_degree")


def scaling_efficiency(throughput_n: float, n: int, throughput_1: float | None) -> float | None:
    if n <= 0 or throughput_n < 0:
        raise ValueError("invalid scaling inputs")
    return None if throughput_1 is None else throughput_n / (n * throughput_1)


def padded_vocab_size(actual_vocab: int, tp_degrees: tuple[int, ...] = (2, 4)) -> int:
    if actual_vocab <= 0 or not tp_degrees or any(degree <= 0 for degree in tp_degrees):
        raise ValueError("vocabulary and TP degrees must be positive")
    divisor = math.lcm(*tp_degrees)
    return math.ceil(actual_vocab / divisor) * divisor


def weak_scaling_metrics(
    update_time_s: float, per_gpu_throughput: float, gpu_count: int
) -> dict[str, float]:
    if update_time_s <= 0 or per_gpu_throughput < 0 or gpu_count <= 0:
        raise ValueError("invalid weak-scaling inputs")
    return {
        "update_time_s": update_time_s,
        "throughput_per_gpu": per_gpu_throughput,
        "aggregate_throughput": per_gpu_throughput * gpu_count,
    }
