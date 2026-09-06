# Execution State

## Post-local-qualification portability hardening

On 2026-09-06, an accidental invocation of the Linux single-node bootstrap from Windows/WSL
exposed a real fail-open portability defect: unverified `python3` could create the Windows-named
environment with Python 3.10 before the heavyweight Torch/CUDA download. This follow-up hardening
does not alter any qualified performance artifact. Commit `0746ebf` and tag
`local-qualified-20260906` remain the qualified local baseline.

All authoritative Linux entry points now share a native-Linux/WSL, required-tool, positive GPU
request, visible-GPU-count, and free-disk preflight. The bootstrap selects Python 3.12, then 3.11,
then another verified Python >=3.11; it validates both the selected interpreter and created
`.venv-linux-core` before pip runs. WSL is fail-closed with `SKIP_UNSUPPORTED` and
`NON_AUTHORITATIVE_PREFLIGHT`, never distributed PASS. The nccl-tests build and runtime remain
locked to `b4d5beebca8a76cf01335f724d154b9b9d394d96`, whose build and JSON argument contracts were
rechecked against NVIDIA's source at that exact revision.

No CUDA model benchmark or distributed qualification ran during this pass. Ruff, mypy, all 45
pytest tests, Git Bash `bash -n`, PowerShell parsing, and sdist/wheel build passed. The accidentally
cross-created `.venv-core` metadata was repaired locally with Windows Python 3.11 without a package
download; its existing Torch 2.14.0+cu132 install remained intact.

Recovered after the 2026-09-06 unexpected shutdown. The recovery audit was read-only until this
state was written. Both the shell location and normalized Git root were
`C:\Users\bjw-0\Downloads\nccl-guard`.

## Crash-recovery findings

- No pre-crash Python or CUDA workload remained. Only the active Codex processes were present.
- `.venv-core` and `.venv-wheel` are intact. Core Python is 3.11.9 with PyTorch 2.14.0+cu132.
- CUDA is available on one NVIDIA GeForce RTX 4090 Laptop GPU (compute capability 8.9); BF16 is
  supported. At recovery, 15,806,234,624 of 17,170,956,288 CUDA-addressable bytes were free.
- All 111 repository evidence JSON files parsed, all six Parquet files were readable, and all 16
  finalized run checksum manifests verified.
- Both profiler traces parsed as JSON. All three local DCP directories contain metadata and a
  complete shard. Dataset manifests and processed files survived.
- No crash-truncated download, partial JSON/Parquet, unfinished finalized run, or damaged
  checkpoint was found.
- The four pre-recovery dense runs are checksum-valid but scientifically excluded: `054448` also
  contaminated timing with gradient inspection, and all four reused one batch while declaring
  unique scheduled identities.
- `artifacts/local-profiler` is a legacy mutable trace and is excluded in favor of immutable
  `artifacts/local-profiler-e79500ea`. These preserved exclusions are not treated as evidence.
- Recovery continuation uncovered a sample-identity mismatch: the three `1803xx` dense blocks and
  `180416` sparse block reused fixed inputs while declaring scheduled identities. They are
  checksum-valid but excluded; benchmark-semantics v2 binds each update to its declared schedule.
- The first resumed E05/E06 simulations materialized an unused full EVALUATION pool too early.
  Those runs are excluded; the corrected protocol creates only selected-config observations after
  the selection freeze and embeds them in immutable run evidence.
- The v1 final-confirmation transaction remains `NOT_STARTED`; recovery does not reopen or alter a
  started transaction.

## Phase classification

| Phase | Recovery class | Basis / next work |
|---:|---|---|
| 0 | COMPLETE_VALID | Root verified and AGENTS.md reread. |
| 1 | COMPLETE_VALID | Doctor evidence and live CUDA recheck agree. |
| 2 | COMPLETE_VALID | Version/source decisions are documented. |
| 3 | COMPLETE_VALID | Installable src-layout package, CLI, configs, and evidence writer exist. |
| 4 | COMPLETE_VALID | Synthetic, WikiText, and bounded C4 artifacts verify; Criteo is a truthful bounded-budget skip. |
| 5 | COMPLETE_VALID | Decoder, RoPE, causal SDPA, SwiGLU, loss, canonical state, and correctness tests exist. |
| 6 | COMPLETE_VALID | Three scheduled-data BF16 launch blocks and H2D evidence passed. |
| 7 | COMPLETE_VALID | Linux NCCL DDP/FSDP2/TP×FSDP code and launchers are syntax/contract tested; execution is externally skipped. |
| 8 | COMPLETE_VALID | Synthetic sparse CUDA passed; TorchRec remains capability-gated and is not called Criteo. |
| 9 | COMPLETE_VALID | Pinned nccl-tests build/run/JSON integration is ready; execution needs Linux multi-GPU. |
| 10 | COMPLETE_VALID | Immutable SHA-covered torch.profiler trace passed; ECR is approximation-only locally. |
| 11 | COMPLETE_VALID | Random/TPE/exhaustive/Guard nested protocol simulation passed without performance claims. |
| 12 | COMPLETE_VALID | Guard budget, visibility, diagnosis and gates are implemented/tested and frozen. |
| 13 | COMPLETE_VALID | Deterministic sample schedule and explicit DCP application state are implemented/tested. |
| 14 | COMPLETE_VALID | Straggler and elastic reference/restart machinery is implemented; execution is externally gated. |
| 15 | COMPLETE_VALID | Locally supported E00/E02/E05/E06 ran; distributed experiments emit truthful skips. |
| 16 | COMPLETE_VALID | Subprocess-isolated TorchTitan config/launch/parser adapter exists; credentials remain external. |
| 17 | COMPLETE_VALID | Local RTX, sparse, profiler, data loader, H2D, checkpoint, and analysis evidence passed. |
| 18 | COMPLETE_VALID | Pre-crash Ruff, mypy, pytest, build, and clean-wheel smoke passed; rerun after changes. |
| 19 | COMPLETE_VALID | Actionable CRITICAL/HIGH/MEDIUM findings were fixed and affected gates rerun. |
| 20 | COMPLETE_VALID | Reports/evidence regenerated from non-excluded SHA-valid runs; verdict remains HOLD. |
| 21 | COMPLETE_VALID | Windows/PowerShell and Linux/Git-Bash syntax checks passed; runbooks give exact commands. |

Current phase: local completion with external distributed gates.

Last successful command: build, clean-wheel CLI/import smoke, and final artifact integrity audit.

Final quality gate: Ruff PASS, mypy PASS across 49 source modules, pytest 45 PASS, sdist/wheel
build PASS, clean-wheel CLI/import smoke PASS, Linux shell syntax PASS, PowerShell syntax PASS.

Closure process audit found and stopped only the two exact Python PIDs orphaned by the interrupted
combined wheel-smoke command. Separate wheel entry-point and import checks passed, and no unrelated
process was touched.

Next command: from Bash on an exclusive native Linux 2/4/8-GPU NCCL host, execute
`GPUS_PER_NODE=<count> bash scripts/linux/bootstrap_single_node.sh`.

Planner freeze: `CREATED`; final confirmation: `NOT_STARTED`.
