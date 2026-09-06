from __future__ import annotations


def diagnose(
    *, ecr: float, hbm_headroom: float, rank_imbalance: float, message_bytes: int, p95_ratio: float
) -> str:
    labels: list[str] = []
    if rank_imbalance >= 0.15:
        labels.append("RANK_IMBALANCED")
    if hbm_headroom <= 0.10:
        labels.append("MEMORY_BOUND")
    if ecr >= 0.30:
        labels.append("COMM_LATENCY_BOUND" if message_bytes < 1 << 20 else "COMM_BANDWIDTH_BOUND")
    if not labels and ecr < 0.10 and p95_ratio < 1.15:
        labels.append("COMPUTE_BOUND")
    return labels[0] if len(labels) == 1 else "MIXED"
