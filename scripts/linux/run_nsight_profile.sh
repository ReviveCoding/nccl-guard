#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "$script_dir/preflight.sh"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"
nccl_guard_authoritative_preflight "$repo_root" "${GPUS_PER_NODE:-}" 5 nsys mkdir
nccl_guard_require_linux_venv "$repo_root"
test "$GPUS_PER_NODE" -ge 2 || { echo 'Need at least two GPUs' >&2; exit 2; }
mkdir -p artifacts/nsight
NCCL_GUARD_STRATEGY="${NCCL_GUARD_STRATEGY:-ddp}" \
nsys profile --trace=cuda,nvtx,nccl,osrt --output=artifacts/nsight/e03 --force-overwrite=false \
  "$repo_root/$NCCL_GUARD_LINUX_VENV/bin/torchrun" --standalone \
  --nproc_per_node="$GPUS_PER_NODE" -m nccl_guard.distributed_worker
