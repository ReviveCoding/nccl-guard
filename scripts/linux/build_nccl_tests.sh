#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "$script_dir/preflight.sh"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"
nccl_guard_authoritative_preflight "$repo_root" "${GPUS_PER_NODE:-}" \
  "${NCCL_GUARD_MIN_FREE_GIB:-5}" git make nproc
revision="${NCCL_TESTS_REVISION:-$NCCL_GUARD_PINNED_NCCL_TESTS_REVISION}"
[[ "$revision" =~ ^[0-9a-f]{40}$ ]] || nccl_guard_error \
  'NCCL_TESTS_REVISION must be an exact lowercase 40-character commit SHA'
[[ "$revision" == "$NCCL_GUARD_PINNED_NCCL_TESTS_REVISION" ]] || nccl_guard_error \
  "NCCL_TESTS_REVISION must remain pinned to $NCCL_GUARD_PINNED_NCCL_TESTS_REVISION"
test -d external/nccl-tests || git clone https://github.com/NVIDIA/nccl-tests.git external/nccl-tests
git -C external/nccl-tests fetch --tags origin
git -C external/nccl-tests checkout --detach "$revision"
test "$(git -C external/nccl-tests rev-parse HEAD)" = "$revision" || nccl_guard_error \
  'checked-out nccl-tests revision does not match the exact pin'
make -C external/nccl-tests -j"$(nproc)" MPI="${MPI:-0}" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
git -C external/nccl-tests rev-parse HEAD > evidence/nccl_tests_commit.txt
