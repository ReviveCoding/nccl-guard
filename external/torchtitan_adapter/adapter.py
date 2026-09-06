"""Subprocess-only TorchTitan adapter; intentionally outside the core package."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def write_config(path: Path, sections: dict[str, dict[str, object]]) -> Path:
    """Write a small deterministic TOML overlay accepted by a pinned TorchTitan checkout."""
    lines: list[str] = []
    for section in sorted(sections):
        lines.append(f"[{section}]")
        for key in sorted(sections[section]):
            value = sections[section][key]
            if isinstance(value, bool):
                encoded = str(value).lower()
            elif isinstance(value, (int, float)):
                encoded = str(value)
            elif isinstance(value, str):
                encoded = json.dumps(value)
            else:
                raise TypeError(f"unsupported TOML value for {section}.{key}")
            lines.append(f"{key} = {encoded}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_metrics(log: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and any(
            key in value for key in ("tokens_per_second", "loss", "step_time")
        ):
            rows.append(value)
    return rows


def launch(
    python: Path, checkout: Path, module: str, config: Path, output_log: Path
) -> subprocess.CompletedProcess[str]:
    command = [str(python), "-m", module, "--config", str(config)]
    result = subprocess.run(
        command, cwd=checkout, capture_output=True, text=True, check=False
    )
    output_log.parent.mkdir(parents=True, exist_ok=True)
    output_log.write_text(result.stdout + result.stderr, encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    args = parser.parse_args()
    result = launch(args.python, args.checkout, args.module, args.config, args.output_log)
    print(json.dumps({"returncode": result.returncode, "metrics": parse_metrics(args.output_log)}))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
