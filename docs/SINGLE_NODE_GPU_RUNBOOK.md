# Single-Node GPU Runbook

Authoritative qualification requires a native Linux host with 2, 4, or 8 homogeneous CUDA GPUs,
a driver supporting CUDA 13.2, at least 25 GiB of free repository-filesystem space, Git, a
compiler, the CUDA toolkit, NCCL, and optional MPI/Nsight. Python 3.12 is preferred; Python 3.11
or another verified Python >=3.11 is accepted. Ensure exclusive GPU allocation.

## Windows PowerShell boundary

Do not invoke `scripts/linux/*.sh` from PowerShell, Git Bash on Windows, WSL, or a Windows-hosted
container. Those contexts cannot produce authoritative NCCL-Guard distributed evidence. From
PowerShell, the qualified local baseline can be verified without moving its tag:

```powershell
git rev-parse 0746ebf
git show-ref --tags local-qualified-20260906
```

Transfer or clone the repository onto the target native Linux host, open Bash there, and change to
the actual checkout directory using its real path.

## RUN THESE ONLY INSIDE THE TARGET LINUX HOST'S BASH SHELL

The environment assignments below are Bash syntax, not PowerShell syntax. Set `GPUS_PER_NODE` to
the exclusively allocated physical GPU count. The bootstrap performs all OS, WSL, interpreter,
tool, disk, and GPU-count checks before creating `.venv-linux-core` or downloading Torch/CUDA
packages.

```bash
GPUS_PER_NODE=8 bash scripts/linux/bootstrap_single_node.sh
GPUS_PER_NODE=8 NCCL_TESTS_REVISION=b4d5beebca8a76cf01335f724d154b9b9d394d96 MPI=0 \
  bash scripts/linux/build_nccl_tests.sh
GPUS_PER_NODE=8 bash scripts/linux/run_nccl_tests.sh
GPUS_PER_NODE=8 bash scripts/linux/run_single_node_8gpu.sh
GPUS_PER_NODE=8 bash scripts/linux/run_fsdp_ablation.sh
GPUS_PER_NODE=8 bash scripts/linux/run_straggler.sh
GPUS_PER_NODE=8 bash scripts/linux/run_recovery.sh
GPUS_PER_NODE=8 NCCL_GUARD_STRATEGY=ddp bash scripts/linux/run_nsight_profile.sh
.venv-linux-core/bin/nccl-guard qualify
```

For a 2- or 4-GPU allocation, replace every `GPUS_PER_NODE=8` assignment consistently. Do not
lower `NCCL_GUARD_MIN_FREE_GIB` unless the operator has separately established that the repository
filesystem can accommodate the environment, nccl-tests build, profiles, checkpoints, and run
artifacts.

Confirm `nvidia-smi topo -m`, NCCL availability, JSON nccl-tests outputs, no thermal/power anomaly,
and SHA-256 manifests. Expected output is immutable `artifacts/<run_id>/` plus small `evidence/`
summaries. A missing NCCL backend, fewer GPUs, OOM, or profiler absence must retain its structured
FAIL/SKIP status; never retry with Gloo as performance evidence.

`run_single_node_8gpu.sh` retains its required historical filename but accepts 2, 4, or 8 GPUs. It
runs ISO-WORKLOAD at one global sample per physical GPU and PRACTICAL_OPTIMIZED at four, preserving
the same global batch across DDP, FSDP2, and each valid TP degree within a track. E10 first creates a
no-fault reference, then performs one elastic restart and compares state/sample/loss fingerprints.
