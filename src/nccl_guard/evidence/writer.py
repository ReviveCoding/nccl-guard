from __future__ import annotations

import json
import uuid
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any


class RunWriter(AbstractContextManager["RunWriter"]):
    def __init__(self, root: Path, kind: str) -> None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.path = root / "artifacts" / f"{stamp}-{kind.lower()}-{uuid.uuid4().hex[:8]}"
        self.path.mkdir(parents=True, exist_ok=False)
        self.finalized = False

    def json(self, name: str, value: Any) -> Path:
        if self.finalized:
            raise RuntimeError("finalized runs are immutable")
        path = self.path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
        return path

    def finalize(self, verdict: dict[str, Any]) -> Path:
        self.json("verdict.json", verdict)
        entries = {}
        for path in sorted(self.path.rglob("*")):
            if path.is_file() and path.name != "sha256_manifest.json":
                entries[path.relative_to(self.path).as_posix()] = sha256(
                    path.read_bytes()
                ).hexdigest()
        manifest = self.json("sha256_manifest.json", entries)
        self.finalized = True
        return manifest

    def __exit__(self, *args: object) -> None:
        if not self.finalized:
            self.finalize({"status": "FAIL", "reason": "writer exited without explicit verdict"})


def verify_run(path: Path) -> bool:
    manifest = json.loads((path / "sha256_manifest.json").read_text(encoding="utf-8"))
    return all(
        (path / name).exists() and sha256((path / name).read_bytes()).hexdigest() == digest
        for name, digest in manifest.items()
    )
