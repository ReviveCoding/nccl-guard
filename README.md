# NCCL-Guard

Topology-Aware Distributed Training Characterization, Configuration Selection, and Reliability
Framework.

NCCL-Guard follows `CHARACTERIZE → DIAGNOSE → TUNE → GUARD → VALIDATE`. It implements a
RoPE decoder LM, DDP/FSDP2/TP×FSDP composition, rank-local overlap metrics, bounded nested search,
checkpoint application state, fault scheduling, immutable evidence, and E00–E12 orchestration.

```powershell
scripts\windows\bootstrap_local.ps1
.\.venv-core\Scripts\nccl-guard.exe doctor
.\.venv-core\Scripts\nccl-guard.exe run-all --profile local-gpu --require-gpu
```

Native Windows results qualify only CUDA single-GPU behavior. NCCL, FSDP2/TP scaling, RDMA, and
multi-node claims require the Linux runbooks. A `SKIP_*` status is never success. The v1 benchmark
verdict vocabulary is `BENCHMARK_QUALIFIED`, `CONDITIONALLY_QUALIFIED`, or `HOLD`.

Raw datasets and run artifacts are ignored. Small claim evidence lives in `evidence/`.
