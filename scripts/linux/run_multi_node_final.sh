#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
for var in NNODES NODE_RANK MASTER_ADDR MASTER_PORT GPUS_PER_NODE NETWORK_INTERFACE; do
  test -n "${!var:-}" || { echo "Missing $var" >&2; exit 2; }
done
export NCCL_SOCKET_IFNAME="$NETWORK_INTERFACE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( $(nproc) / GPUS_PER_NODE ))}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
for strategy in ddp fsdp2 tp_fsdp; do
  NCCL_GUARD_STRATEGY="$strategy" torchrun --nnodes="$NNODES" --node_rank="$NODE_RANK" \
    --nproc_per_node="$GPUS_PER_NODE" --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" -m nccl_guard.distributed_worker
done
