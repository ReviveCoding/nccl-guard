#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "$script_dir/preflight.sh"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"
for var in NNODES NODE_RANK MASTER_ADDR MASTER_PORT GPUS_PER_NODE NETWORK_INTERFACE; do
  test -n "${!var:-}" || { echo "Missing $var" >&2; exit 2; }
done
nccl_guard_authoritative_preflight "$repo_root" "$GPUS_PER_NODE" \
  "${NCCL_GUARD_MIN_FREE_GIB:-5}" ip
nccl_guard_require_linux_venv "$repo_root"
test "$NNODES" -ge 2 || { echo 'NNODES must be >=2' >&2; exit 2; }
test "$NODE_RANK" -lt "$NNODES" || { echo 'NODE_RANK out of range' >&2; exit 2; }
ip link show "$NETWORK_INTERFACE" >/dev/null
nvidia-smi -L
"$NCCL_GUARD_LINUX_VENV/bin/nccl-guard" doctor
