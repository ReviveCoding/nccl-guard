#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
gpus="${GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}"
test "$gpus" -ge 2 || { echo 'Need at least two GPUs' >&2; exit 2; }
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( $(nproc) / gpus ))}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
NCCL_GUARD_STRATEGY=fsdp2 NCCL_GUARD_FSDP_ROOT_ONLY=1 \
  torchrun --standalone --nproc_per_node="$gpus" -m nccl_guard.distributed_worker
NCCL_GUARD_STRATEGY=fsdp2 torchrun --standalone --nproc_per_node="$gpus" \
  -m nccl_guard.distributed_worker
NCCL_GUARD_STRATEGY=fsdp2 NCCL_GUARD_RESHARD_AFTER_FORWARD=0 \
  torchrun --standalone --nproc_per_node="$gpus" -m nccl_guard.distributed_worker
NCCL_GUARD_STRATEGY=fsdp2 NCCL_GUARD_SEPARATE_RS=1 \
  torchrun --standalone --nproc_per_node="$gpus" -m nccl_guard.distributed_worker
