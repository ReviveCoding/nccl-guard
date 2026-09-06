#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "$script_dir/preflight.sh"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"
gpus="${GPUS_PER_NODE:-}"
nccl_guard_authoritative_preflight "$repo_root" "$gpus" 5 nproc date
nccl_guard_require_linux_venv "$repo_root"
test "$gpus" -ge 2 || { echo 'Need at least two GPUs' >&2; exit 2; }
torchrun="$repo_root/$NCCL_GUARD_LINUX_VENV/bin/torchrun"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
reference_output="artifacts/${stamp}-e10-reference"
reference_checkpoint="artifacts/${stamp}-e10-reference-checkpoint"
recovery_output="artifacts/${stamp}-e10-recovery"
recovery_checkpoint="artifacts/${stamp}-e10-recovery-checkpoint"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( $(nproc) / gpus ))}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
NCCL_GUARD_RECOVERY_OUTPUT="$reference_output" \
NCCL_GUARD_RECOVERY_CHECKPOINT="$reference_checkpoint" NCCL_GUARD_INJECT_FAILURE=0 \
  "$torchrun" --standalone --nproc_per_node="$gpus" --max_restarts=0 \
  -m nccl_guard.recovery_worker
export NCCL_GUARD_RECOVERY_OUTPUT="$recovery_output"
export NCCL_GUARD_RECOVERY_CHECKPOINT="$recovery_checkpoint"
export NCCL_GUARD_INJECT_FAILURE=1
"$torchrun" --standalone --nproc_per_node="$gpus" --max_restarts=1 \
  --monitor_interval=1 -m nccl_guard.recovery_worker
"$NCCL_GUARD_LINUX_VENV/bin/python" -m nccl_guard.recovery.compare \
  --reference "$reference_output" --recovered "$recovery_output"
