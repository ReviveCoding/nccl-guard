# Claim Ledger

| Claim | State | Evidence boundary |
|---|---|---|
| CUDA executes on local RTX GPU | SUPPORTED | hardware capabilities and E02 run |
| BF16 dense scheduled-data training is finite | SUPPORTED | three benchmark-semantics v2 blocks |
| BF16 synthetic sparse control training is finite | SUPPORTED | local sparse CUDA artifact; not Criteo |
| RoPE causal decoder and deterministic data contracts | SUPPORTED | unit tests |
| DDP/FSDP2/TP composition implemented | CONDITIONAL | source/control tests; Linux runtime pending |
| Nested SEARCH/EVALUATION protocol isolation | SUPPORTED | deterministic offline simulation/tests only |
| Local DCP application-state restore | SUPPORTED | model/optimizer/scheduler/RNG/step/cursor utility check |
| NCCL/DDP/FSDP2/TP performance | UNSUPPORTED | native Windows, one GPU |
| RDMA/InfiniBand/RoCE/EFA | UNSUPPORTED | no fabric evidence |
| Fault recovery | UNSUPPORTED | E10 needs Linux multi-GPU execution |
| Guard near-exhaustive held-out performance | UNSUPPORTED | no distributed E05 evaluation pool |
| Bounded C4 acquisition/pipeline | SUPPORTED | 100-row hash-verified subset; no performance claim |
| Criteo systems results | UNSUPPORTED | bounded viewer failed; full corpus rejected by budget |

Current overall verdict: **HOLD**. Local CUDA qualification is supported; implementation evidence is
not distributed benchmark qualification.
