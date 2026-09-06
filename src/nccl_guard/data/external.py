from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from nccl_guard.data.core import DatasetManifest
from nccl_guard.status import Status

DATASETS = {
    "wikitext103": {
        "dataset": "Salesforce/wikitext",
        "config": "wikitext-103-raw-v1",
        "split": "test",
        "license": "CC BY-SA (dataset metadata says 3.0; current card text says 4.0)",
        "rows": 100,
    },
    "c4": {
        "dataset": "allenai/c4",
        "config": "en",
        "split": "validation",
        "license": "ODC-BY plus Common Crawl terms",
        "rows": 100,
    },
    "criteo": {
        "dataset": "criteo/CriteoClickLogs",
        "config": "criteo",
        "split": "train",
        "license": "CC BY-NC-SA 4.0",
        "rows": 100,
    },
}


def _bounded_get(url: str, max_bytes: int) -> bytes:
    request = Request(url, headers={"User-Agent": "nccl-guard/0.1 bounded-research-client"})
    with urlopen(request, timeout=60) as response:  # noqa: S310 - fixed HTTPS endpoint
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > max_bytes:
            raise MemoryError(f"response {length} exceeds {max_bytes}-byte cap")
        body = response.read(max_bytes + 1)
    if len(body) > max_bytes:
        raise MemoryError(f"response exceeds {max_bytes}-byte cap")
    return body


def acquire_viewer_subset(root: Path, name: str, max_bytes: int = 50 << 20) -> tuple[Status, Path]:
    spec = DATASETS[name]
    query = urlencode(
        {
            "dataset": spec["dataset"],
            "config": spec["config"],
            "split": spec["split"],
            "offset": 0,
            "length": spec["rows"],
        }
    )
    url = f"https://datasets-server.huggingface.co/rows?{query}"
    manifest_path = root / "data/manifests" / f"{name}.json"
    try:
        raw = _bounded_get(url, max_bytes)
        payload: dict[str, Any] = json.loads(raw)
        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError("dataset viewer returned no rows")
        normalized = [item.get("row", item) for item in rows]
        output = root / "data/processed" / f"{name}_viewer_subset.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        encoded = "".join(json.dumps(row, sort_keys=True) + "\n" for row in normalized).encode()
        output.write_bytes(encoded)
        tokens = sum(
            len(str(row.get("text", "")).split()) for row in normalized if isinstance(row, dict)
        )
        manifest = DatasetManifest(
            name,
            str(spec["dataset"]),
            "main/viewer-offset-0",
            str(spec["license"]),
            len(encoded),
            len(normalized),
            tokens,
            sha256(encoded).hexdigest(),
        )
        manifest.write(manifest_path)
        return Status.PASS, manifest_path
    except MemoryError as exc:
        status = Status.SKIP_RESOURCE_BUDGET
        reason = str(exc)
    except Exception as exc:
        status = Status.SKIP_RESOURCE_BUDGET if name == "criteo" else Status.SKIP_UNSUPPORTED
        reason = repr(exc)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "name": name,
                "source": spec["dataset"],
                "license": spec["license"],
                "status": status,
                "reason": reason,
                "max_response_bytes": max_bytes,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return status, manifest_path


def prepare_external(root: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name in DATASETS:
        status, path = acquire_viewer_subset(root, name)
        result[name] = {"status": str(status), "manifest": str(path)}
    return result


def verify_external(root: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for name in DATASETS:
        manifest_path = root / "data/manifests" / f"{name}.json"
        if not manifest_path.exists():
            results[name] = {"status": Status.FAIL, "reason": "manifest missing"}
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(manifest.get("status", "")).startswith("SKIP_"):
            results[name] = {
                "status": manifest["status"],
                "reason": manifest.get("reason"),
                "verified": False,
            }
            continue
        data_path = root / "data/processed" / f"{name}_viewer_subset.jsonl"
        valid = (
            data_path.is_file()
            and data_path.stat().st_size == manifest.get("bytes")
            and sha256(data_path.read_bytes()).hexdigest() == manifest.get("sha256")
        )
        results[name] = {
            "status": Status.PASS if valid else Status.FAIL,
            "verified": valid,
            "bytes": data_path.stat().st_size if data_path.is_file() else 0,
            "rows": manifest.get("rows", 0),
            "tokens": manifest.get("tokens", 0),
        }
    return results
