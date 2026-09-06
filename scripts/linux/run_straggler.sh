#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "$script_dir/preflight.sh"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"
gpus="${GPUS_PER_NODE:-}"
nccl_guard_authoritative_preflight "$repo_root" "$gpus" 1 nproc
nccl_guard_require_linux_venv "$repo_root"
test "$gpus" -ge 2 || { echo 'Need at least two GPUs' >&2; exit 2; }
torchrun="$repo_root/$NCCL_GUARD_LINUX_VENV/bin/torchrun"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( $(nproc) / gpus ))}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
NCCL_GUARD_STRATEGY=ddp "$torchrun" --standalone --nproc_per_node="$gpus" \
  -m nccl_guard.distributed_worker
NCCL_GUARD_STRATEGY=ddp NCCL_GUARD_FAULT_KIND=compute_delay NCCL_GUARD_FAULT_STEP=4 \
NCCL_GUARD_FAULT_RANK=1 NCCL_GUARD_FAULT_DURATION_MS=100 \
  "$torchrun" --standalone --nproc_per_node="$gpus" -m nccl_guard.distributed_worker
NCCL_GUARD_STRATEGY=ddp NCCL_GUARD_FAULT_KIND=workload_skew NCCL_GUARD_FAULT_STEP=4 \
NCCL_GUARD_FAULT_RANK=1 NCCL_GUARD_SKEW_REPEATS=2 \
  "$torchrun" --standalone --nproc_per_node="$gpus" -m nccl_guard.distributed_worker
