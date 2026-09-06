from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import blake2b, sha256
from pathlib import Path

import numpy as np


def stable_bucket(value: str, buckets: int, key: bytes = b"nccl-guard-v1") -> int:
    if buckets <= 0:
        raise ValueError("buckets must be positive")
    digest = blake2b(value.encode(), key=key, digest_size=8).digest()
    return int.from_bytes(digest, "little") % buckets


@dataclass(frozen=True)
class DatasetManifest:
    name: str
    source: str
    revision: str
    license: str
    bytes: int
    rows: int
    tokens: int
    sha256: str
    transform_version: str = "v1"
    seed: int = 17

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")


def prepare_synthetic(
    root: Path, rows: int = 4096, seq: int = 128, vocab: int = 8192
) -> DatasetManifest:
    path = root / "data" / "processed" / "synthetic_dense_v1.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(17)
    values = rng.integers(0, vocab, size=(rows, seq + 1), dtype=np.uint16)
    values.tofile(path)
    digest = sha256(path.read_bytes()).hexdigest()
    manifest = DatasetManifest(
        "synthetic-dense",
        "deterministic generator",
        "v1",
        "CC0",
        path.stat().st_size,
        rows,
        rows * (seq + 1),
        digest,
    )
    manifest.write(root / "data" / "manifests" / "synthetic_dense.json")
    return manifest


def verify_manifest(root: Path, manifest_path: Path) -> bool:
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_path = root / "data" / "processed" / "synthetic_dense_v1.bin"
    return data_path.exists() and sha256(data_path.read_bytes()).hexdigest() == record["sha256"]


@dataclass(frozen=True)
class SampleSchedule:
    seed: int
    updates: int
    global_batch: int

    def ids(self) -> list[list[int]]:
        start = self.seed * 1_000_000
        return [
            [start + u * self.global_batch + i for i in range(self.global_batch)]
            for u in range(self.updates)
        ]

    def for_dp_rank(self, rank: int, dp_degree: int) -> list[list[int]]:
        if rank < 0 or rank >= dp_degree:
            raise ValueError("invalid DP rank")
        return [row[rank::dp_degree] for row in self.ids()]

    def fingerprint(self) -> str:
        return sha256(json.dumps(self.ids(), separators=(",", ":")).encode()).hexdigest()


def verify_no_duplicate_or_skip(schedule: SampleSchedule, dp_degree: int) -> bool:
    reconstructed: list[list[int]] = [[] for _ in range(schedule.updates)]
    for rank in range(dp_degree):
        for update, ids in enumerate(schedule.for_dp_rank(rank, dp_degree)):
            reconstructed[update].extend(ids)
    return all(
        sorted(got) == sorted(expected)
        for got, expected in zip(reconstructed, schedule.ids(), strict=True)
    )
