# Fault and Recovery

Fault events support compute delay, workload skew, and rank termination. Stragglers use a rolling
median plus scaled MAD. Linux hard-failure qualification uses torchrun elastic restart and PyTorch
Distributed Checkpoint state-dict APIs.

Model and optimizer are distributed state. Application state explicitly includes scheduler, global
step, Python/NumPy/Torch/CUDA RNG, global data cursor, dataset fingerprint, and config fingerprint.
A restart is not a PASS: same-world-size E10 must verify subsequent state and loss trajectory.
World-size-changing duplicate/skip claims require actual sample-ID remapping verification.

`scripts/linux/run_recovery.sh` launches a no-fault reference and a one-restart fault transaction.
The comparison gates model, optimizer, scheduler and RNG fingerprints, step/cursor, exact consumed
sample IDs, and subsequent loss. This path is implemented and syntax-tested but remains
`SKIP_UNSUPPORTED` locally because Windows has no NCCL backend or second GPU.
