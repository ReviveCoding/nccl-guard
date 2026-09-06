#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -m venv .venv-core
.venv-core/bin/pip install --upgrade pip
.venv-core/bin/pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu132
.venv-core/bin/pip install -e '.[dev,tuning,data]'
.venv-core/bin/nccl-guard doctor
