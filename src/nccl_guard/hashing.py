"""Bounded-memory hashing for dataset, checkpoint, and evidence files."""

from hashlib import file_digest
from pathlib import Path


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()
