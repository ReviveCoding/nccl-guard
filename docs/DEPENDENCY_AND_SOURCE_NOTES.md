# Dependency and Source Notes

Verified 2026-09-06 against official upstream sources:

- PyTorch 2.14.0 stable, CUDA 13.2 Windows wheel, CPython 3.11 locally. CPython 3.12/3.13 were
  installed through Microsoft Store but both failed `ensurepip`; 3.11 was the newest operational
  interpreter. <https://pytorch.org/blog/pytorch-2-14-release-blog/>
- FSDP2 uses `torch.distributed.fsdp.fully_shard`; TP uses `parallelize_module` with a 1-D mesh;
  both remain experimental. <https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html>
  <https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html>
- DCP uses `get_state_dict`, `set_state_dict`, `save`, and `load`; arbitrary application state is
  explicit. <https://docs.pytorch.org/docs/stable/distributed.checkpoint.html>
- nccl-tests upstream supports `-J` JSON, cycles, correctness, warmup, and timeout. The cluster build
  records its exact commit. The 2026-09-06 cluster handoff pins upstream master as observed then,
  `b4d5beebca8a76cf01335f724d154b9b9d394d96`. <https://github.com/NVIDIA/nccl-tests>
- NCCL RAS JSON requires NCCL 2.28.7 or later; older installs are not replaced merely for RAS.
  <https://github.com/NVIDIA/nccl/blob/master/docs/userguide/source/troubleshooting/ras.rst>
- Nsight Systems advanced NCCL tracing requires compatible recent NCCL and is optional.
  <https://docs.nvidia.com/nsight-systems/UserGuide/>
- TorchRec/FBGEMM and TorchTitan are isolated capability-gated environments; no compatible Windows
  stack was assumed. TorchComms is a 2.13+ emerging path and is gated by import and execution.
  TorchRec S3 uses the official collective planner call. TorchTitan currently recommends nightly
  for main and exposes `--module`/`--config`; it remains separate. TorchComms enablement uses
  `torch.distributed.config.use_torchcomms` or `TORCH_DISTRIBUTED_USE_TORCHCOMMS=1` when supported.

Local resolved core: torch 2.14.0+cu132, CUDA runtime 13.2, driver 610.62, NumPy 2.4.6, pandas
3.0.5, PyArrow 25.0.1. The native Windows wheel reports Gloo but not NCCL; Gloo is not qualification.
