# Metrics

Dense throughput is processed input tokens per second, where processed tokens equal global samples
times sequence length. Strong scaling fixes global work; weak scaling fixes per-DP-rank work.
One-GPU efficiency is omitted when no feasible one-GPU denominator exists.

For each rank, communication and compute intervals are independently unioned. Exposed communication
is `duration(C \ K)` and ECR divides it by update duration. Nsight evidence is `ECR_AUTH`;
torch.profiler is `ECR_APPROX`. Durations are never summed across ranks. Trainer communication
efficiency is comparable only for identical collective, ranks, datatype, message-size regime, node
count, and topology.
