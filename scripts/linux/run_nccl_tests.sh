#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
test "$(uname -s)" = Linux || { echo 'nccl-tests requires Linux' >&2; exit 2; }
for tool in nvidia-smi timeout; do
  command -v "$tool" >/dev/null || { echo "Missing $tool" >&2; exit 2; }
done
gpus="${GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}"
case "$gpus" in 2|4|8) ;; *) echo 'GPUS_PER_NODE must be 2, 4, or 8' >&2; exit 2 ;; esac
revision="$(git -C external/nccl-tests rev-parse HEAD)"
test -n "$revision"
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)"
test "$free_mib" -gt 0
mkdir -p artifacts/nccl evidence
for operation in all_reduce all_gather reduce_scatter alltoall; do
  output="artifacts/nccl/${operation}-${gpus}gpu.json"
  .venv-core/bin/python -c \
    "from pathlib import Path; from nccl_guard.collectives.nccl_tests import command; import subprocess; raise SystemExit(subprocess.run(command('$operation', Path('$output'), $((free_mib * 1024 * 1024)), $gpus)).returncode)"
done
printf '%s\n' "$revision" > evidence/nccl_tests_commit.txt
