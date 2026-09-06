#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
gpus="${GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}"
case "$gpus" in 2|4|8) ;; *) echo 'GPUS_PER_NODE must be 2, 4, or 8' >&2; exit 2 ;; esac
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( $(nproc) / gpus ))}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
NCCL_GUARD_TRACK=ISO_WORKLOAD NCCL_GUARD_STRATEGY=ddp NCCL_GUARD_MICROBATCH=1 \
NCCL_GUARD_GRAD_ACCUMULATION=1 torchrun --standalone --nproc_per_node="$gpus" \
  -m nccl_guard.distributed_worker
NCCL_GUARD_TRACK=ISO_WORKLOAD NCCL_GUARD_STRATEGY=fsdp2 NCCL_GUARD_MICROBATCH=1 \
NCCL_GUARD_GRAD_ACCUMULATION=1 torchrun --standalone --nproc_per_node="$gpus" \
  -m nccl_guard.distributed_worker
tp_degrees=(2)
test "$gpus" -lt 4 || tp_degrees+=(4)
for tp in "${tp_degrees[@]}"; do
  NCCL_GUARD_TRACK=ISO_WORKLOAD NCCL_GUARD_STRATEGY=tp_fsdp NCCL_GUARD_TP_DEGREE="$tp" \
  NCCL_GUARD_MICROBATCH=1 NCCL_GUARD_GRAD_ACCUMULATION="$tp" \
    torchrun --standalone --nproc_per_node="$gpus" -m nccl_guard.distributed_worker
done
# Practical configs retain 4*gpus global samples/update across strategies.
NCCL_GUARD_TRACK=PRACTICAL_OPTIMIZED NCCL_GUARD_STRATEGY=ddp NCCL_GUARD_MICROBATCH=4 \
NCCL_GUARD_GRAD_ACCUMULATION=1 torchrun --standalone --nproc_per_node="$gpus" \
  -m nccl_guard.distributed_worker
NCCL_GUARD_TRACK=PRACTICAL_OPTIMIZED NCCL_GUARD_STRATEGY=fsdp2 NCCL_GUARD_MICROBATCH=2 \
NCCL_GUARD_GRAD_ACCUMULATION=2 torchrun --standalone --nproc_per_node="$gpus" \
  -m nccl_guard.distributed_worker
for tp in "${tp_degrees[@]}"; do
  accumulation="$(( 4 * tp / 2 ))"
  NCCL_GUARD_TRACK=PRACTICAL_OPTIMIZED NCCL_GUARD_STRATEGY=tp_fsdp \
  NCCL_GUARD_TP_DEGREE="$tp" NCCL_GUARD_MICROBATCH=2 \
  NCCL_GUARD_GRAD_ACCUMULATION="$accumulation" \
    torchrun --standalone --nproc_per_node="$gpus" -m nccl_guard.distributed_worker
done
