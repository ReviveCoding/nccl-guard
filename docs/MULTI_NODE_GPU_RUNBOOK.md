# Multi-Node GPU Runbook

Provisioning is external. Use homogeneous native Linux nodes, a shared code/config revision,
synchronized clocks, a reachable rendezvous address, identical NCCL/CUDA/PyTorch, exclusive GPUs,
and verified IB/RDMA/RoCE only when the actual fabric supports it. Run nccl-tests with MPI before
trainers. WSL, Git Bash on Windows, and Windows-hosted containers are not qualification hosts.

## Windows PowerShell boundary

Do not paste Bash environment assignments into PowerShell and do not invoke the Linux scripts from
Windows. PowerShell may be used only to verify the preserved baseline before transferring the
repository:

```powershell
git rev-parse 0746ebf
git show-ref --tags local-qualified-20260906
```

On every target node, open Bash and change to that node's actual repository checkout directory.

## RUN THESE ONLY INSIDE THE TARGET LINUX HOST'S BASH SHELL

These `NAME=value bash ...` and `export` forms are Bash syntax, not PowerShell syntax. On every node,
first create the distinct Linux environment and then validate the multi-node settings. Substitute
the real rank, rank-0 address, and verified interface for that node:

```bash
GPUS_PER_NODE=8 bash scripts/linux/bootstrap_single_node.sh
export NNODES=2
export NODE_RANK=<0-or-1>
export MASTER_ADDR=<rank0-ip-address>
export MASTER_PORT=29500
export GPUS_PER_NODE=8
export NETWORK_INTERFACE=<verified-interface-name>
bash scripts/linux/bootstrap_multi_node.sh
bash scripts/linux/run_multi_node_final.sh
```

On rank 0, also build the exact pinned nccl-tests commit and coordinate MPI nccl-tests across the
same hosts before training:

```bash
GPUS_PER_NODE=8 NCCL_TESTS_REVISION=b4d5beebca8a76cf01335f724d154b9b9d394d96 MPI=1 \
  bash scripts/linux/build_nccl_tests.sh
```

Store checkpoint paths on storage visible to all ranks for multi-node E10; the provided
`run_recovery.sh` is the same-node qualification entry point.

Freeze the planner and reach PREFLIGHT_PASSED before STARTED. After STARTED, do not change workload,
thresholds, candidate space, datasets, or metrics. Infrastructure invalidation becomes
TERMINAL_FAILURE and a new versioned transaction. Collect at least five independent launch blocks
when budget permits, alternating A/B order. Verify run hashes before qualification.
