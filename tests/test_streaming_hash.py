import hashlib
import json
from pathlib import Path

import pytest

from nccl_guard.evidence.writer import RunWriter, verify_run
from nccl_guard.hashing import file_sha256
from nccl_guard.recovery.compare import compare


@pytest.mark.parametrize("size", [0, 13, 2 * 1024 * 1024 + 17])
def test_file_hash_streams_without_read_bytes(tmp_path: Path, monkeypatch, size: int) -> None:
    data = bytes(range(256)) * (size // 256) + bytes(range(size % 256))
    path = tmp_path / "checkpoint.bin"
    path.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()

    def forbidden_read_bytes(self: Path) -> bytes:
        raise AssertionError("file-sized reads are forbidden")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read_bytes)
    assert file_sha256(path) == expected


def test_evidence_writer_retains_hashes_and_tamper_detection(tmp_path: Path, monkeypatch) -> None:
    with RunWriter(tmp_path, "streaming") as writer:
        payload = writer.path / "checkpoint.bin"
        payload.write_bytes(b"checkpoint" * 200_000)
        monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail("unbounded read"))
        manifest_path = writer.finalize({"status": "PASS"})
    manifest = json.loads(manifest_path.read_text())
    assert manifest["checkpoint.bin"] == hashlib.sha256(b"checkpoint" * 200_000).hexdigest()
    assert verify_run(writer.path)
    payload.write_bytes(b"tampered")
    assert not verify_run(writer.path)


def test_recovery_comparison_hashes_large_files_without_read_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    reference, recovered = tmp_path / "reference", tmp_path / "recovered"
    reference.mkdir()
    recovered.mkdir()
    for root in (reference, recovered):
        (root / "recovery.json").write_text(json.dumps({"ranks": []}))
    content = b"checkpoint" * 200_000
    (recovered / "checkpoint.bin").write_bytes(content)
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail("unbounded read"))
    assert compare(reference, recovered)["status"] == "FAIL"
    manifest = json.loads((recovered / "sha256_manifest.json").read_text())
    assert manifest["checkpoint.bin"] == hashlib.sha256(content).hexdigest()
