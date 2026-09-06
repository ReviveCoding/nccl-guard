# Experiment Design

E00 qualifies environment/data. E01 characterizes NCCL. E02 is the single-GPU reference. E03 has
ISO-WORKLOAD and PRACTICAL-OPTIMIZED tracks. E04 studies FSDP2 F0–F3. E05 compares a strong fixed
default, Random, TPE, exhaustive-search reference, and Guard. E06 ablates Guard inputs. E07/E08 are
frozen unseen-workload/topology confirmations. E09/E10 test stragglers/recovery. E11/E12 isolate
TorchTitan/TorchComms.

SEARCH observations select every method. Selected configurations alone are measured on the
independent EVALUATION pool. Individual steps are repeated measurements, not independent samples;
the inferential unit is an independent launch block. Paired A/B blocks alternate AB/BA. Compilation
is disabled for v1 headline comparisons. Final confirmation uses CREATED, PREFLIGHT_PASSED,
STARTED, COMPLETED, or TERMINAL_FAILURE and cannot mutate v1 after STARTED.
