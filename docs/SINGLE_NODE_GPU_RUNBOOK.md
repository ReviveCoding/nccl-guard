# Single-Node GPU Runbook

Prerequisites: Linux, 2/4/8 homogeneous CUDA GPUs, driver supporting CUDA 13.2, Python 3.11+, Git,
compiler, CUDA toolkit, NCCL, and optional MPI/Nsight. Ensure exclusive GPU allocation.

```bash
bash scripts/linux/bootstrap_single_node.sh
NCCL_TESTS_REVISION=b4d5beebca8a76cf01335f724d154b9b9d394d96 MPI=0 \
  bash scripts/linux/build_nccl_tests.sh
GPUS_PER_NODE=<2-or-4-or-8> bash scripts/linux/run_nccl_tests.sh
GPUS_PER_NODE=<2-or-4-or-8> bash scripts/linux/run_single_node_8gpu.sh
GPUS_PER_NODE=<2-or-4-or-8> bash scripts/linux/run_fsdp_ablation.sh
GPUS_PER_NODE=<2-or-4-or-8> bash scripts/linux/run_straggler.sh
GPUS_PER_NODE=<2-or-4-or-8> bash scripts/linux/run_recovery.sh
GPUS_PER_NODE=<2-or-4-or-8> NCCL_GUARD_STRATEGY=ddp bash scripts/linux/run_nsight_profile.sh
.venv-core/bin/nccl-guard qualify
```

Confirm `nvidia-smi topo -m`, NCCL availability, JSON nccl-tests outputs, no thermal/power anomaly,
and SHA-256 manifests. Expected output is immutable `artifacts/<run_id>/` plus small `evidence/`
summaries. A missing NCCL backend, fewer GPUs, OOM, or profiler absence must retain its structured
FAIL/SKIP status; never retry with Gloo as performance evidence.

`run_single_node_8gpu.sh` retains its required historical filename but accepts 2, 4, or 8 GPUs. It
runs ISO-WORKLOAD at one global sample per physical GPU and PRACTICAL_OPTIMIZED at four, preserving
the same global batch across DDP, FSDP2, and each valid TP degree within a track. E10 first creates a
no-fault reference, then performs one elastic restart and compares state/sample/loss fingerprints.
