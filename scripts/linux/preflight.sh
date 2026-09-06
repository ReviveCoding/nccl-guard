#!/usr/bin/env bash

# Shared fail-closed checks for authoritative Linux qualification entry points.
NCCL_GUARD_LINUX_VENV=".venv-linux-core"
NCCL_GUARD_PINNED_NCCL_TESTS_REVISION="b4d5beebca8a76cf01335f724d154b9b9d394d96"

nccl_guard_error() {
  printf 'INVALID_CONFIGURATION: %s\n' "$*" >&2
  return 2
}

nccl_guard_require_tools() {
  local tool
  for tool in "$@"; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      nccl_guard_error \
        "required tool '$tool' was not found; install it on the target Linux host"
      return
    fi
  done
}

nccl_guard_require_authoritative_linux() {
  local kernel release osrelease=""
  kernel="$(uname -s 2>/dev/null || true)"
  if [[ "$kernel" != "Linux" ]]; then
    nccl_guard_error \
      "authoritative qualification requires native Linux; detected '${kernel:-unknown}'"
    return
  fi

  release="$(uname -r 2>/dev/null || true)"
  if [[ -r /proc/sys/kernel/osrelease ]]; then
    IFS= read -r osrelease < /proc/sys/kernel/osrelease || true
  fi
  if [[ "${release,,}" == *microsoft* ]] || [[ "${osrelease,,}" == *microsoft* ]] || \
     [[ "${osrelease,,}" == *wsl* ]]; then
    printf '%s\n' \
      'SKIP_UNSUPPORTED: WSL is not an authoritative NCCL-Guard qualification host.' \
      'NON_AUTHORITATIVE_PREFLIGHT: use a native Linux multi-GPU host; no distributed PASS was emitted.' \
      >&2
    return 2
  fi
}

nccl_guard_select_python() {
  local candidate version
  local -a candidates=(python3.12 python3.11 python3 python)
  for candidate in "${candidates[@]}"; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    if version="$("$candidate" -c \
      'import sys; print(".".join(map(str, sys.version_info[:3]))); raise SystemExit(sys.version_info < (3, 11))' \
      2>/dev/null)"; then
      printf 'Python preflight accepted %s (%s).\n' "$candidate" "$version" >&2
      printf '%s\n' "$candidate"
      return 0
    fi
    version="$("$candidate" --version 2>&1 || true)"
    printf 'Python preflight rejected %s (%s); Python >=3.11 is required.\n' \
      "$candidate" "${version:-unknown version}" >&2
  done
  nccl_guard_error \
    'no compatible Python found; install python3.12 or python3.11 and rerun before installing dependencies'
}

nccl_guard_validate_gpu_request() {
  local requested="$1" listing visible masked_visible
  if [[ ! "$requested" =~ ^[1-9][0-9]*$ ]]; then
    nccl_guard_error 'GPUS_PER_NODE must be set to a positive integer'
    return
  fi
  if ! listing="$(nvidia-smi -L 2>/dev/null)"; then
    nccl_guard_error 'nvidia-smi could not enumerate GPUs on the target Linux host'
    return
  fi
  visible="$(awk '/^GPU [0-9]+:/ { count++ } END { print count + 0 }' <<<"$listing")"
  if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    if [[ -z "$CUDA_VISIBLE_DEVICES" ]] || [[ "$CUDA_VISIBLE_DEVICES" == "-1" ]]; then
      visible=0
    else
      masked_visible="$(awk -F, \
        '{ for (i = 1; i <= NF; i++) if ($i != "") count++ } END { print count + 0 }' \
        <<<"$CUDA_VISIBLE_DEVICES")"
      if (( masked_visible < visible )); then
        visible="$masked_visible"
      fi
    fi
  fi
  if (( visible < requested )); then
    nccl_guard_error \
      "visible GPU count ($visible) is smaller than GPUS_PER_NODE ($requested)"
    return
  fi
  printf 'GPU preflight accepted %s requested / %s visible.\n' "$requested" "$visible" >&2
}

nccl_guard_validate_disk() {
  local path="$1" minimum_gib="$2" available_kib required_kib
  if [[ ! "$minimum_gib" =~ ^[1-9][0-9]*$ ]]; then
    nccl_guard_error 'minimum free disk requirement must be a positive integer GiB value'
    return
  fi
  if ! available_kib="$(df -Pk "$path" | awk 'NR == 2 { print $4 }')" || \
     [[ ! "$available_kib" =~ ^[0-9]+$ ]]; then
    nccl_guard_error "could not determine free disk space for $path"
    return
  fi
  required_kib=$((minimum_gib * 1024 * 1024))
  if (( available_kib < required_kib )); then
    nccl_guard_error \
      "insufficient disk space: need ${minimum_gib} GiB free at $path before continuing"
    return
  fi
}

nccl_guard_require_linux_venv() {
  local root="$1"
  if [[ ! -x "$root/$NCCL_GUARD_LINUX_VENV/bin/python" ]]; then
    nccl_guard_error \
      "Linux environment missing; run GPUS_PER_NODE=<count> bash scripts/linux/bootstrap_single_node.sh first"
    return
  fi
}

nccl_guard_authoritative_preflight() {
  local root="$1" requested="$2" minimum_gib="$3"
  shift 3
  nccl_guard_require_authoritative_linux || return
  nccl_guard_require_tools uname awk df nvidia-smi "$@" || return
  nccl_guard_validate_gpu_request "$requested" || return
  nccl_guard_validate_disk "$root" "$minimum_gib" || return
}
