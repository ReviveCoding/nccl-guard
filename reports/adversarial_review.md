# Adversarial Review

Status: completed after local execution. No unresolved actionable CRITICAL or HIGH findings.

| Severity | Finding | Resolution |
|---|---|---|
| HIGH | Gradient correctness work entered the CUDA timing region. | Moved all finite-gradient and norm inspection after the end event; reran E02. |
| HIGH | Linux scripts wrote control metadata without launching NCCL trainers. | Added a real torchrun/NCCL worker for DDP, FSDP2, and TP×FSDP with rank evidence. |
| HIGH | TP worker initially changed global batch relative to DDP. | TP-degree accumulation restores identical world-size global batch; regression tested. |
| HIGH | TorchRec planner used an unverified call. | Matched current official `collective_plan(model, sharders, pg)` and retained TorchRec as S3. |
| HIGH | Profiler reused a mutable fixed path. | Every profiler execution now has a UUID path. |
| HIGH | `qualify` could return PASS without distributed primary evidence. | It now returns nonzero with benchmark verdict HOLD. |
| MEDIUM | `run-all` omitted local profiler/loader/checkpoint/data attempts. | Added them and executed the corrected command. |
| MEDIUM | FSDP2 F3 was not connected to the current API. | Added capability-gated `set_separate_reduce_scatter_group`. |
| MEDIUM | Criteo viewer failure was classified unsupported. | Reclassified `SKIP_RESOURCE_BUDGET`; direct corpus files exceed bounded intent. |
| LOW | Deprecated `pynvml` metapackage emitted warnings. | Removed it; retained the maintained `nvidia-ml-py` provider. |
| CRITICAL | Generic Linux experiment gates could return PASS without launching distributed work. | Gates now FAIL on capable hosts and direct operators to explicit NCCL/torchrun launchers. |
| HIGH | Dense/sparse runs reused one batch while declaring unique scheduled IDs. | Bound every update to its declared schedule; excluded affected runs and reran benchmark-semantics v2. |
| HIGH | Offline simulation constructed the EVALUATION table before selections froze. | EVALUATION observations are now created only for frozen selections; excluded and reran E05/E06. |
| HIGH | DCP loaded stored application fields without restoring scheduler or runtime RNGs. | Added active scheduler/Python/NumPy/Torch/CUDA RNG restore and fingerprint gates. |
| HIGH | E10 had helpers but no executable elastic fault/reference path. | Added Linux FSDP2 DCP worker, controlled rank exit, restart, and reference comparison. |
| HIGH | E03 practical configs did not execute true microbatches. | Worker now batches scheduled IDs and scripts preserve global batch through DP-only arithmetic. |
| MEDIUM | Utility checkpoint/profiler artifacts lacked SHA manifests. | Added manifests and reran both utilities. |
| MEDIUM | Nsight wrapper profiled a metadata gate and nccl-tests defaulted to `master`. | Nsight now wraps torchrun; nccl-tests requires an exact revision. |
| MEDIUM | Doctor conflated importability with execution capability. | Package entries now record `importable` and `execution_verified` separately. |
| MEDIUM | TPE repeated suggestions could be counted as new configuration evaluations. | Budget now counts unique configurations over one stable Optuna space. |
| HIGH | Linux bootstrap could select Python 3.10, reuse the Windows-named venv, and begin multi-GB dependency installation before rejecting the project. | Added a shared fail-closed native-Linux preflight, ordered Python 3.12/3.11/verified fallback selection, pre-install version/GPU/tool/disk checks, WSL rejection, and `.venv-linux-core` separation. |
| HIGH | Linux launchers could enter authoritative qualification from WSL or a Windows Bash context and could run an unpinned nccl-tests checkout. | Every Linux build/launch entry point now shares the authoritative-host guard; WSL emits `SKIP_UNSUPPORTED` and `NON_AUTHORITATIVE_PREFLIGHT`, while build and runtime enforce the exact pinned commit. |

Audited: root boundary, CUDA fallback, batch arithmetic, strong/weak definitions, TP divisibility,
TP-before-FSDP2, bottom-up FSDP2, optimizer order, canonical state, RoPE/causal mask/vocabulary,
SEARCH/EVALUATION isolation, planner visibility/freeze, exhaustive-reference wording, tuning costs,
rank-local ECR union, collective comparability, launch-level statistics, thermal state, exclusive
authoritative runs, DCP application state, sample remapping, bounded downloads/licenses, TorchRec
fairness, TorchTitan isolation, SKIP semantics, hashes, portability, and docs.

Post-local-qualification portability review preserved commit `0746ebf` and tag
`local-qualified-20260906`; no qualified performance evidence or planner scientific configuration
was changed, and no CUDA/distributed benchmark was rerun. Regression coverage exercises Python
3.10 rejection, Python >=3.11 acceptance, WSL rejection, GPU-count mismatch, Linux venv isolation,
the exact nccl-tests pin, and shell/runbook consistency.

Remaining MEDIUM external limitations: no Linux multi-GPU/NCCL execution, no Nsight ECR_AUTH, no
executed elastic E10, no Criteo bounded rows, no TorchRec/TorchTitan/TorchComms compatible local
stack, and no real independent distributed EVALUATION pool. These are explicitly unsupported and
cannot be repaired on this machine without external hardware/software/credentials.
