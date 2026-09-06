# Multi-Node GPU Runbook

Provisioning is external. Use homogeneous Linux nodes, shared code/config revision, synchronized
clocks, reachable rendezvous address, identical NCCL/CUDA/PyTorch, exclusive GPUs, and verified
IB/RDMA/RoCE only when the actual fabric supports it. Run nccl-tests with MPI before trainers.

On every node set:

```bash
export NNODES=2 NODE_RANK=<0-or-1> MASTER_ADDR=<rank0-ip> MASTER_PORT=29500
export GPUS_PER_NODE=8 NETWORK_INTERFACE=<verified-interface>
bash scripts/linux/bootstrap_multi_node.sh
bash scripts/linux/run_multi_node_final.sh
```

On rank 0, also build the exact pinned nccl-tests revision and coordinate MPI nccl-tests across the
same hosts before training. Store checkpoint paths on storage visible to all ranks for multi-node
E10; the provided `run_recovery.sh` is the same-node qualification entry point.

Freeze the planner and reach PREFLIGHT_PASSED before STARTED. After STARTED, do not change workload,
thresholds, candidate space, datasets, or metrics. Infrastructure invalidation becomes
TERMINAL_FAILURE and a new versioned transaction. Collect at least five independent launch blocks
when budget permits, alternating A/B order. Verify run hashes before qualification.
