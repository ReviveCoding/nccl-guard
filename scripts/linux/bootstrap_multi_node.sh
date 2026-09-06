#!/usr/bin/env bash
set -euo pipefail
for var in NNODES NODE_RANK MASTER_ADDR MASTER_PORT GPUS_PER_NODE NETWORK_INTERFACE; do
  test -n "${!var:-}" || { echo "Missing $var" >&2; exit 2; }
done
test "$NNODES" -ge 2 || { echo 'NNODES must be >=2' >&2; exit 2; }
test "$NODE_RANK" -lt "$NNODES" || { echo 'NODE_RANK out of range' >&2; exit 2; }
ip link show "$NETWORK_INTERFACE" >/dev/null
nvidia-smi -L
.venv-core/bin/nccl-guard doctor
