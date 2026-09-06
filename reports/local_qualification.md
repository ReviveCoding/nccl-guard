# Local Qualification

Status: **PASS for bounded local CUDA qualification; HOLD for the distributed benchmark.**

Machine: Windows, NVIDIA GeForce RTX 4090 Laptop GPU. PyTorch 2.14.0+cu132 reports CUDA 13.2 and
BF16 support. NCCL is unavailable, and Gloo was not substituted.

Dense benchmark-semantics v2 used the persisted deterministic dataset through scheduled sample
IDs, pinned host memory, nonblocking H2D outside model timing, and 10 warmup plus 50 measured
updates per launch. Across 3 independent run blocks, input-token
throughput was 10438.88 mean and 10603.67 median tokens/s
(run-level bootstrap 95% interval 10025.06–10687.92).
This is single-configuration characterization, not an inferential method comparison.

The synthetic schema-compatible sparse CUDA control run `20260906T180718Z-sparse-844527b9` achieved
8004.12 samples/s over 20 measured updates,
with bf16 precision and 105463808 peak allocated bytes.
It is not Criteo or distributed TorchRec evidence.

The existing torch.profiler trace is labeled ECR_APPROX. DCP local utility evidence validates
model/optimizer/application-state restore but is not elastic rank-failure recovery.
