#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "$script_dir/preflight.sh"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

nccl_guard_require_authoritative_linux
python_bin="$(nccl_guard_select_python)"
nccl_guard_require_tools uname awk df nvidia-smi
nccl_guard_validate_gpu_request "${GPUS_PER_NODE:-}"
nccl_guard_validate_disk "$repo_root" "${NCCL_GUARD_MIN_FREE_GIB:-25}"
if [[ -d "$NCCL_GUARD_LINUX_VENV/Scripts" ]]; then
  nccl_guard_error \
    "$NCCL_GUARD_LINUX_VENV has Windows layout; remove it and rerun on the native Linux host"
fi

"$python_bin" -m venv "$NCCL_GUARD_LINUX_VENV"
"$NCCL_GUARD_LINUX_VENV/bin/python" -c \
  'import sys; raise SystemExit(sys.version_info < (3, 11))' || nccl_guard_error \
  'created Linux environment is not Python >=3.11; remove it and install python3.12 or python3.11'
"$NCCL_GUARD_LINUX_VENV/bin/python" -m pip install --upgrade pip
"$NCCL_GUARD_LINUX_VENV/bin/python" -m pip install \
  torch==2.14.0 --index-url https://download.pytorch.org/whl/cu132
"$NCCL_GUARD_LINUX_VENV/bin/python" -m pip install -e '.[dev,tuning,data]'
"$NCCL_GUARD_LINUX_VENV/bin/nccl-guard" doctor
