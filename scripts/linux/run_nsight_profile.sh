#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
command -v nsys >/dev/null || { echo 'Nsight Systems unavailable' >&2; exit 2; }
test "$(uname -s)" = Linux || { echo 'NCCL profiling requires Linux' >&2; exit 2; }
test -n "${GPUS_PER_NODE:-}" || { echo 'Set GPUS_PER_NODE' >&2; exit 2; }
test "$GPUS_PER_NODE" -ge 2 || { echo 'Need at least two GPUs' >&2; exit 2; }
mkdir -p artifacts/nsight
NCCL_GUARD_STRATEGY="${NCCL_GUARD_STRATEGY:-ddp}" \
nsys profile --trace=cuda,nvtx,nccl,osrt --output=artifacts/nsight/e03 --force-overwrite=false \
  torchrun --standalone --nproc_per_node="$GPUS_PER_NODE" -m nccl_guard.distributed_worker
