#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "$script_dir/preflight.sh"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"
gpus="${GPUS_PER_NODE:-}"
nccl_guard_authoritative_preflight "$repo_root" "$gpus" \
  "${NCCL_GUARD_MIN_FREE_GIB:-5}" git timeout sort head mkdir
nccl_guard_require_linux_venv "$repo_root"
case "$gpus" in 2|4|8) ;; *) echo 'GPUS_PER_NODE must be 2, 4, or 8' >&2; exit 2 ;; esac
[[ -d external/nccl-tests/.git ]] || nccl_guard_error \
  'nccl-tests is not built; run scripts/linux/build_nccl_tests.sh first'
revision="$(git -C external/nccl-tests rev-parse HEAD)"
[[ "$revision" == "$NCCL_GUARD_PINNED_NCCL_TESTS_REVISION" ]] || nccl_guard_error \
  "nccl-tests must be checked out at $NCCL_GUARD_PINNED_NCCL_TESTS_REVISION"
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)"
test "$free_mib" -gt 0
mkdir -p artifacts/nccl evidence
for operation in all_reduce all_gather reduce_scatter alltoall; do
  output="artifacts/nccl/${operation}-${gpus}gpu.json"
  "$NCCL_GUARD_LINUX_VENV/bin/python" -c \
    "from pathlib import Path; from nccl_guard.collectives.nccl_tests import command; import subprocess; raise SystemExit(subprocess.run(command('$operation', Path('$output'), $((free_mib * 1024 * 1024)), $gpus)).returncode)"
done
printf '%s\n' "$revision" > evidence/nccl_tests_commit.txt
