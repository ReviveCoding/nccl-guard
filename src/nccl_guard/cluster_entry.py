from __future__ import annotations

import argparse
from pathlib import Path

from nccl_guard.experiments.runner import run_experiment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()
    experiment = "E08" if args.profile == "multi-node-final" else "E03"
    status, _ = run_experiment(Path.cwd(), experiment)
    return 0 if status.is_success else 2


if __name__ == "__main__":
    raise SystemExit(main())
