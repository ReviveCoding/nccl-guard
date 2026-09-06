# NCCL-Guard Repository Instructions

## Mission
Implement NCCL-Guard as an evidence-backed distributed-training
characterization, tuning, and reliability research framework.

Implement and test rather than merely planning.

## Machine and safety boundary
- Work inside this repository.
- Do not modify drivers, BIOS, system CUDA, WSL, or unrelated files.
- Do not provision paid cloud resources.
- Do not push or publish unless explicitly requested later.
- Prefer repository-local environments and caches.
- Use network/package permissions only when required.

## Windows boundary
Native Windows may provide real local CUDA single-GPU evidence.
Do not claim NCCL, DDP/FSDP/TP scaling, RDMA, or multi-node performance
from native Windows execution.

Do not substitute Gloo results for NCCL qualification.

## Scientific integrity
Never fabricate distributed evidence.

Use explicit statuses including:
- PASS
- FAIL
- SKIP_UNSUPPORTED
- SKIP_NEEDS_CREDENTIALS
- SKIP_VERSION_MISMATCH
- INVALID_CONFIGURATION

A skipped experiment is never a successful experiment.

## Distributed correctness
- Apply TP before FSDP2 when composing them.
- Apply FSDP2 bottom-up to Transformer blocks and then the root.
- Construct the optimizer after FSDP2 sharding.
- Validate TP divisibility before execution.
- Compute global batch/tokens from DATA-PARALLEL degree, not total world size.
- Separate strong scaling from weak scaling.
- Do not calculate 1-GPU scaling efficiency for a model that does not fit on one GPU.

## Experimental fairness
- Use the same canonical initial model state and data identity across comparable strategies.
- Keep correctness-mode and performance-mode separate.
- Keep torch.compile disabled for v1 headline comparisons.
- Do not expose the complete oracle table to NCCL-Guard.
- Separate tuning/search observations from untouched final evaluation observations.
- Include NCCL-Guard characterization/profile cost in tuning-cost reporting.
- Never tune v1 after frozen final workload/topology results are opened.

## Metrics
Do not sum NCCL durations across ranks to estimate exposed communication.

ECR must use timeline interval overlap on a rank-local basis and clearly
distinguish authoritative Nsight evidence from profiler approximations.

Trainer Communication Efficiency is valid only when collective type,
rank count, datatype, message-size regime, node count, and topology are comparable.

## Checkpointing
Use PyTorch Distributed Checkpoint state-dict APIs for distributed model
and optimizer state.

Explicitly checkpoint application state such as scheduler, step, RNG,
and deterministic global data cursor.

Do not claim duplicate-free or skip-free world-size-changing recovery
unless the global sample-ID remapping is actually implemented and verified.

## Data
Use bounded public data only.

Do not download full C4 or full large Criteo corpora.
Record source revision, terms/license, hashes, transformations, and row/token counts.
Do not silently substitute another dataset when credentials or terms block access.

## Review loop
Before completion:
1. inspect current official APIs for all version-sensitive paths,
2. run formatting/lint/type/tests/package checks,
3. run every genuinely supported local experiment,
4. inspect git status/diff and generated evidence,
5. adversarially review numerical correctness, distributed semantics,
   leakage, benchmark fairness, portability, and claim boundaries,
6. fix actionable issues and rerun affected checks.

Final benchmark verdict vocabulary:
BENCHMARK_QUALIFIED
CONDITIONALLY_QUALIFIED
HOLD
