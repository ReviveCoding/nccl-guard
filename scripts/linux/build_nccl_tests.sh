#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
test -n "${NCCL_TESTS_REVISION:-}" || {
  echo 'Set NCCL_TESTS_REVISION to an exact commit or tag' >&2
  exit 2
}
revision="$NCCL_TESTS_REVISION"
test -d external/nccl-tests || git clone https://github.com/NVIDIA/nccl-tests.git external/nccl-tests
git -C external/nccl-tests fetch --tags origin
git -C external/nccl-tests checkout --detach "$revision"
make -C external/nccl-tests -j"$(nproc)" MPI="${MPI:-0}" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
git -C external/nccl-tests rev-parse HEAD > evidence/nccl_tests_commit.txt
