#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "$script_dir/preflight.sh"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"
for var in NNODES NODE_RANK MASTER_ADDR MASTER_PORT GPUS_PER_NODE NETWORK_INTERFACE; do
  test -n "${!var:-}" || { echo "Missing $var" >&2; exit 2; }
done
nccl_guard_authoritative_preflight "$repo_root" "$GPUS_PER_NODE" 5 nproc ip
nccl_guard_require_linux_venv "$repo_root"
test "$NNODES" -ge 2 || { echo 'NNODES must be >=2' >&2; exit 2; }
test "$NODE_RANK" -lt "$NNODES" || { echo 'NODE_RANK out of range' >&2; exit 2; }
ip link show "$NETWORK_INTERFACE" >/dev/null
torchrun="$repo_root/$NCCL_GUARD_LINUX_VENV/bin/torchrun"
export NCCL_SOCKET_IFNAME="$NETWORK_INTERFACE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( $(nproc) / GPUS_PER_NODE ))}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
for strategy in ddp fsdp2 tp_fsdp; do
  NCCL_GUARD_STRATEGY="$strategy" "$torchrun" --nnodes="$NNODES" --node_rank="$NODE_RANK" \
    --nproc_per_node="$GPUS_PER_NODE" --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" -m nccl_guard.distributed_worker
done
