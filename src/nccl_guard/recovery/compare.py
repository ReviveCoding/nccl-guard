from __future__ import annotations

import argparse
import json
import math
from hashlib import sha256
from pathlib import Path


def compare(reference: Path, recovered: Path) -> dict[str, object]:
    ref = json.loads((reference / "recovery.json").read_text(encoding="utf-8"))
    got = json.loads((recovered / "recovery.json").read_text(encoding="utf-8"))
    ref_ranks = {row["rank"]: row for row in ref["ranks"]}
    got_ranks = {row["rank"]: row for row in got["ranks"]}
    rank_checks: dict[str, bool] = {}
    for rank, expected in ref_ranks.items():
        actual = got_ranks.get(rank, {})
        rank_checks[str(rank)] = (
            actual.get("global_step") == expected["global_step"]
            and actual.get("data_cursor") == expected["data_cursor"]
            and actual.get("consumed_sample_ids") == expected["consumed_sample_ids"]
            and actual.get("model_fingerprint") == expected["model_fingerprint"]
            and actual.get("optimizer_fingerprint") == expected["optimizer_fingerprint"]
            and actual.get("scheduler_fingerprint") == expected["scheduler_fingerprint"]
            and actual.get("rng_fingerprint") == expected["rng_fingerprint"]
            and math.isclose(
                float(actual.get("last_loss", math.inf)),
                float(expected["last_loss"]),
                rel_tol=1e-6,
                abs_tol=1e-7,
            )
        )
    result: dict[str, object] = {
        "status": "PASS" if rank_checks and all(rank_checks.values()) else "FAIL",
        "rank_checks": rank_checks,
        "verified": [
            "global_step",
            "data_cursor",
            "sample_ids_no_duplicate_or_skip",
            "model_state_fingerprint",
            "optimizer_state_fingerprint",
            "scheduler_state_fingerprint",
            "rng_state_fingerprint",
            "subsequent_loss",
        ],
        "world_size_changing_recovery": "NOT_VERIFIED",
    }
    comparison = recovered / "reference_comparison.json"
    comparison.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    hashes = {
        item.name: sha256(item.read_bytes()).hexdigest()
        for item in sorted(recovered.iterdir())
        if item.is_file() and item.name != "sha256_manifest.json"
    }
    (recovered / "sha256_manifest.json").write_text(
        json.dumps(hashes, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--recovered", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.recovered)
    print(json.dumps(result))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
