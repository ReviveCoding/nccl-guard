# Model and Parallelism

The decoder is token embedding → repeated pre-norm attention/SwiGLU blocks → final RMSNorm → LM
projection. Q/K receive RoPE and attention uses PyTorch SDPA with `is_causal=True`. The loss is
next-token cross entropy. Padded vocabulary and hidden/head/FFN dimensions are validated against TP.

For a 2-D mesh, `world_size = dp_degree × tp_degree`, but global batch is
`microbatch_per_DP_rank × accumulation × dp_degree`; TP does not duplicate samples. TP is applied
first, FSDP2 is applied blocks-bottom-up then root, and the optimizer is constructed last.
